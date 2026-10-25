import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

if __package__:
    from .tracker import PlanarTracker
else:
    from tracker import PlanarTracker


def track_video(video_path, output_dir, label="Object", show=False, roi=None):
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Не удалось прочитать видео: {video_path}")

        # Если задана область, фоновая часть первого кадра не входит в эталон.
        cv2.setRNGSeed(0)
        tracker = PlanarTracker(frame, roi=roi)
        height, width = frame.shape[:2]

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = {0, total // 3, 2 * total // 3, max(0, total - 1)}
        samples = []
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / f"{video_path.stem}_tracked.avi"
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"MJPG"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Не удалось создать видео: {output_video}")

        frames = 0
        modes = dict(detect=0, flow=0, hold=0, lost=0)
        colors = dict(detect=(0, 255, 0), flow=(0, 220, 255), hold=(0, 100, 255))
        start = perf_counter()
        while ok:
            polygon, mode = tracker.update(frame)
            modes[mode] += 1
            if polygon is not None:
                color = colors[mode]
                cv2.polylines(frame, [np.int32(polygon)], True, color, 2,
                              cv2.LINE_AA)
                x, y = polygon.reshape(-1, 2).min(axis=0).astype(int)
                position = (int(np.clip(x, 0, width - 1)),
                            int(np.clip(y - 8, 20, height - 1)))
                cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 1, cv2.LINE_AA)

            info = f"Frame {frames} | {mode}"
            cv2.putText(frame, info, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, info, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if frames in sample_indices:
                samples.append(frame.copy())
            frames += 1
            if show:
                cv2.imshow("Tracking (Esc to exit)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            ok, frame = capture.read()

        elapsed = perf_counter() - start
        if samples:
            while len(samples) < 4:
                samples.append(samples[-1].copy())
            preview = np.vstack([np.hstack(samples[:2]), np.hstack(samples[2:4])])
            success, data = cv2.imencode(".jpg", preview)
            if not success:
                raise ValueError("Не удалось сохранить примеры кадров")
            data.tofile(output_dir / f"{video_path.stem}_preview.jpg")
        stats = {
            "video": video_path.name,
            "roi": roi,
            "frames": frames,
            "tracked_frames": frames - modes["lost"],
            "tracking_percent": round(100 * (frames - modes["lost"]) / frames, 2),
            "modes": modes,
            "processing_seconds": round(elapsed, 2),
            "processing_fps": round(frames / elapsed, 2),
        }
        (output_dir / f"{video_path.stem}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return stats
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Трекинг объекта по ключевым точкам SIFT")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--label", default="Object", help="Подпись латиницей")
    parser.add_argument("--show", action="store_true", help="Показывать обработку в окне")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Область объекта на первом кадре, в пикселях")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Файл не найден: {args.video}")
    try:
        stats = track_video(args.video, args.output, args.label, args.show, args.roi)
    except ValueError as error:
        parser.error(str(error))
    print(f"Обработано кадров: {stats['frames']}")
    print(f"Кадров с рамкой: {stats['tracked_frames']} ({stats['tracking_percent']}%)")
    print(f"Режимы: {stats['modes']}")
    print(f"Скорость обработки: {stats['processing_fps']} кадров/с")
    print(f"Результаты: {args.output}")


if __name__ == "__main__":
    main()
