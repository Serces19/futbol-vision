#!/usr/bin/env python3
"""
ANÁLISIS COMPLETO DE VIDEO DE FÚTBOL
Script único que hace TODO el análisis y genera 3 videos de salida:
1. Video anotado con jugadores, IDs, equipos y coordenadas del campo
2. Video debug con líneas detectadas para verificar calibración
3. Video mapa táctico 2D con vista desde arriba

USO: python analisis_completo_video.py
"""

import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

# Configurar un logger más detallado para debug
log_level = logging.DEBUG # Cambiar a logging.INFO para menos detalle
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from football_analytics.core.config import ConfigManager
    from football_analytics.detection.object_detector import ObjectDetector
    from football_analytics.tracking.player_tracker import PlayerTracker
    from football_analytics.classification.team_classifier import TeamClassifier
    
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'PnLCalib'))
    from PnLCalib.inference import get_camera_params_for_frame, projection_from_cam_params, project, inference
    from PnLCalib.model.cls_hrnet import get_cls_net
    from PnLCalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from PnLCalib.utils.utils_calib import FramebyFrameCalib

    logger.info("✅ Módulos importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando módulos: {e}")
    sys.exit(1)

# --- Funciones de Calibración ---

def get_homography_from_params(params_dict):
    """
    Calcula la homografía de Mundo->Imagen, consistente con la lógica de PnLCalib.
    PnLCalib usa un sistema de coordenadas con origen en el centro del campo.
    """
    if not params_dict or 'cam_params' not in params_dict:
        return None
    try:
        cam_params = params_dict['cam_params']
        x_focal, y_focal = cam_params['x_focal_length'], cam_params['y_focal_length']
        px, py = cam_params['principal_point']
        R = np.array(cam_params['rotation_matrix'], dtype=np.float64)
        
        # 'position_meters' es la posición de la cámara C en el mundo.
        # El vector de traslación t para la matriz extrínseca es -R @ C.
        C = np.array(cam_params['position_meters'], dtype=np.float64).reshape(3, 1)
        t = -R @ C

        K = np.array([[x_focal, 0, px], [0, y_focal, py], [0, 0, 1]], dtype=np.float64)
        
        # El plano del campo es XY (Z es vertical). Usamos la 1ra (X) y 2da (Y) columna de R.
        extrinsic_3x3 = np.hstack((R[:, [0, 1]], t))
        
        H = K @ extrinsic_3x3
        return H
    except (KeyError, TypeError) as e:
        logger.warning(f"Advertencia al calcular homografía: {e}")
        return None

def get_field_position(calibration_state, pixel_point):
    """
    Transforma un punto de la imagen a coordenadas del campo (con origen en esquina).
    """
    if not calibration_state.get('is_ready') or calibration_state.get('inverse_homography') is None:
        return None
    try:
        pixel_point_np = np.array([pixel_point], dtype=np.float32).reshape(1, 1, 2)

        # La homografía inversa mapea de la imagen a un sistema de coordenadas del mundo
        # con el origen en el CENTRO del campo (ej. X: [-52.5, 52.5], Y: [-34, 34]).
        field_point_centered = cv2.perspectiveTransform(pixel_point_np, calibration_state['inverse_homography'])
        
        if field_point_centered is None:
            return None
        
        # Ajustamos el punto al sistema de coordenadas que espera el resto de la aplicación,
        # con el origen en la ESQUINA superior izquierda (ej. X: [0, 105], Y: [0, 68]).
        field_len, field_w = get_field_dimensions()
        x_adj = field_point_centered[0][0][0] + field_len / 2
        y_adj = field_point_centered[0][0][1] + field_w / 2
        
        return (x_adj, y_adj)
    except cv2.error:
        logger.warning("Error de OpenCV en perspectiveTransform, probablemente por una homografía inválida.")
        return None

def is_calibrator_ready(calibration_state):
    return calibration_state.get('is_ready', False)

def get_field_dimensions():
    return 105.0, 68.0

# --- Funciones de Dibujo ---

def dibujar_anotaciones(frame, tracked_objects, ball_position, calibration_state):
    team_colors = {0: (255, 100, 100), 1: (100, 100, 255), None: (128, 128, 128)}
    for obj in tracked_objects:
        center_point = ((obj.detection.bbox[0] + obj.detection.bbox[2]) // 2, obj.detection.bbox[3])
        x1, y1, x2, y2 = obj.detection.bbox
        color = team_colors.get(obj.team_id, team_colors[None])
        #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        field_pos = getattr(obj, 'field_pos', None)
        team_name = "" if obj.team_id == 0 else "" if obj.team_id == 1 else ""
        label = f"ID:{obj.track_id} {team_name}"
        if field_pos:
            label += f" ({field_pos[0]:.1f},{field_pos[1]:.1f}m)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1) 
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        overlay = frame.copy()
        cv2.ellipse(overlay, center_point, axes=(15, 5), angle=0, startAngle=0, endAngle=360, color=color, thickness=3)
        alpha = 0.4  # transparencia
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    calib_text = "CALIBRADO" if is_calibrator_ready(calibration_state) else "SIN CALIBRAR"
    calib_color = (0, 255, 0) if is_calibrator_ready(calibration_state) else (0, 0, 255)
    cv2.putText(frame, calib_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 2)
    return frame


def dibujar_debug_lineas(frame, calibration_state):
    """Dibuja la visualización de depuración de la calibración."""
    debug_frame = frame.copy()

    # Dibuja las esquinas del campo proyectadas si la calibración es exitosa
    if is_calibrator_ready(calibration_state) and calibration_state.get('homography') is not None:
        H = calibration_state['homography']
        field_len, field_w = get_field_dimensions()
        
        # Coordenadas de las esquinas del campo con origen en la esquina (0,0)
        # para que coincida con la homografía calculada.
        corners_field = np.array([
            [[0, 0]],
            [[field_len, 0]],
            [[field_len, field_w]],
            [[0, field_w]],
        ], dtype=np.float32)

        # Proyectar las esquinas a la imagen
        projected_corners = cv2.perspectiveTransform(corners_field, H)
        
        if projected_corners is not None:
            # Dibujar polígono y círculos en las esquinas
            cv2.polylines(debug_frame, [np.int32(projected_corners)], isClosed=True, color=(0, 255, 255), thickness=2)
            for i, corner in enumerate(projected_corners):
                pt = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(debug_frame, pt, 10, (0, 0, 255), -1)
                cv2.putText(debug_frame, f"C{i}", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    params = calibration_state.get('params')
    camera_info = ""
    if params and 'cam_params' in params:
        cam_params = params['cam_params']
        pan, tilt = cam_params.get('pan_degrees', 0), cam_params.get('tilt_degrees', 0)
        camera_info = f"Pan: {pan:.1f} deg, Tilt: {tilt:.1f} deg"
    info_lines = [
        "Debug PnLCalib",
        f"Estado: {'✅ CALIBRADO' if is_calibrator_ready(calibration_state) else '❌ SIN CALIBRAR'}",
        camera_info
    ]
    for i, info in enumerate(info_lines):
        if info: cv2.putText(debug_frame, info, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return debug_frame

def crear_mapa_2d(width=800, height=600):
    field_map = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
    margin = 50
    field_w_px, field_h_px = width - 2 * margin, height - 2 * margin
    cv2.rectangle(field_map, (margin, margin), (width - margin, height - margin), (255, 255, 255), 2)
    cv2.line(field_map, (width // 2, margin), (width // 2, height - margin), (255, 255, 255), 2)
    cv2.circle(field_map, (width // 2, height // 2), int(9.15 * (field_w_px / 105.0)), (255, 255, 255), 2)
    return field_map, margin, field_w_px, field_h_px

def campo_a_mapa(field_pos, map_dims):
    if field_pos is None: return None
    field_len, field_w = get_field_dimensions()
    map_w, map_h, margin, field_w_px, field_h_px = map_dims
    x_map = margin + (field_pos[0] / field_len) * field_w_px
    y_map = margin + (field_pos[1] / field_w) * field_h_px
    if not (margin <= x_map < map_w - margin and margin <= y_map < map_h - margin):
        logger.debug(f"Coordenada de campo {field_pos} está fuera de los límites del mapa.")
        return None
    return (int(x_map), int(y_map))

def dibujar_mapa_tactico(tracked_objects, calibration_state, frame_count, map_width=800, map_height=600):
    field_map, margin, field_w_px, field_h_px = crear_mapa_2d(map_width, map_height)
    map_dims = (map_width, map_height, margin, field_w_px, field_h_px)
    team_colors = {0: (255, 0, 0), 1: (0, 0, 255), None: (128, 128, 128)}
    for obj in tracked_objects:
        map_pos = campo_a_mapa(getattr(obj, 'field_pos', None), map_dims)
        if map_pos:
            color = team_colors.get(obj.team_id, team_colors[None])
            cv2.circle(field_map, map_pos, 8, color, -1)
            cv2.putText(field_map, str(obj.track_id), (map_pos[0] + 10, map_pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    info = f"Frame: {frame_count} | Calibrado: {'SI' if is_calibrator_ready(calibration_state) else 'NO'}"
    cv2.putText(field_map, info, (10, map_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return field_map

# --- Función Principal ---

def main():
    logger.info("🏈 ANÁLISIS COMPLETO DE VIDEO DE FÚTBOL")
    video_path = "./dataset_curado/video3.mp4"
    if not Path(video_path).exists():
        logger.error(f"❌ Video no encontrado: {video_path}"); return

    try:
        logger.info("1️⃣ Inicializando componentes...")
        config = ConfigManager()
        detector = ObjectDetector(model_path="models/best_v02.pt", confidence_threshold=0.3)
        tracker = PlayerTracker(config=config.tracker_config)
        team_classifier = TeamClassifier(config=config.processing_config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {device}")
        
        with open("PnLCalib/config/hrnetv2_w48.yaml", 'r') as f: cfg = yaml.safe_load(f)
        with open("PnLCalib/config/hrnetv2_w48_l.yaml", 'r') as f: cfg_l = yaml.safe_load(f)
        
        kp_model = get_cls_net(cfg)
        kp_model.load_state_dict(torch.load("models/SV_FT_WC14_kp", map_location=device))
        kp_model.to(device).eval()

        line_model = get_cls_net_l(cfg_l)
        line_model.load_state_dict(torch.load("models/SV_FT_WC14_lines", map_location=device))
        line_model.to(device).eval()
        logger.info("✅ Modelos de calibración cargados.")

        cap = cv2.VideoCapture(video_path)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_anotado = cv2.VideoWriter("./output/video_anotado.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out_debug = cv2.VideoWriter("./output/video_debug_lineas.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out_tactico = cv2.VideoWriter("./output/video_mapa_tactico.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 600))

        cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
        calibration_state = {'is_ready': False}

        logger.info("3️⃣ Procesando video...")
        frame_count = 0
        max_frames = int(fps * 5) # Procesar 5 segundos

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break

            try:
                detections = detector.detect(frame)
                tracked_objects = tracker.update(detections, frame.shape[:2])
                if len(tracked_objects) > 1: tracked_objects = team_classifier.classify_teams(tracked_objects, frame)

                params_dict = get_camera_params_for_frame(
                    frame, kp_model, line_model, 
                    kp_threshold=0.3, line_threshold=0.3, pnl_refine=False, cam=cam, device=device
                )
                
                if params_dict:
                    P = projection_from_cam_params(params_dict)
                    projected_frame = project(frame, P)
                    H = get_homography_from_params(params_dict)
                    if H is not None:
                        logger.debug(f"Frame {frame_count}: Homografía calculada:\\n{H}")
                        calibration_state.update({
                            'homography': H,
                            'inverse_homography': np.linalg.inv(H),
                            'params': params_dict, 
                            'is_ready': True
                        })
                        if frame_count % int(fps) == 0: logger.info(f"Frame {frame_count}: ✅ Calibración exitosa.")
                    else:
                        calibration_state['is_ready'] = False
                else:
                    calibration_state['is_ready'] = False

                for i, obj in enumerate(tracked_objects):
                    # Usar el punto inferior central del bounding box para una mejor correspondencia con el suelo
                    center_point = ((obj.detection.bbox[0] + obj.detection.bbox[2]) // 2, obj.detection.bbox[3])
                    obj.field_pos = get_field_position(calibration_state, center_point)
                    if i == 0 and obj.field_pos: # Log solo para el primer jugador y si la posición es válida
                        logger.debug(f"Frame {frame_count}: Jugador {obj.track_id} | Pos Imagen: {center_point} -> Pos Campo: ({obj.field_pos[0]:.2f}, {obj.field_pos[1]:.2f})")
                ball_pos = next((((d.bbox[0] + d.bbox[2]) // 2, (d.bbox[1] + d.bbox[3]) // 2) for d in detections if d.class_name == 'ball'), None)

                out_anotado.write(dibujar_anotaciones(frame.copy(), tracked_objects, ball_pos, calibration_state))
                #out_debug.write(dibujar_debug_lineas(frame.copy(), calibration_state))
                out_debug.write(projected_frame)
                out_tactico.write(dibujar_mapa_tactico(tracked_objects, calibration_state, frame_count))

            except Exception as e:
                logger.error(f"❌ Error en frame {frame_count}: {e}", exc_info=True)
            
            frame_count += 1
            cv2.imshow("Video Anotado", dibujar_anotaciones(frame.copy(), tracked_objects, ball_pos, calibration_state))
            cv2.imshow("Video Anotado", dibujar_mapa_tactico(tracked_objects, calibration_state, frame_count))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()
        cap.release()
        [o.release() for o in [out_anotado, out_debug, out_tactico]]
        logger.info("🎉 ANÁLISIS TERMINADO")

    except Exception as e:
        logger.error(f"❌ Error fatal en la ejecución: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()