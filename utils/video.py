import cv2


class VideoReadError(Exception):
    """To Be raised if video cannot be read."""


def read_video(video_path, frame_indices: list[int] | None = None):
    if frame_indices == []:
        raise ValueError("Frame Indices cannot be empty list")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(f"Error Opening Stream @ {video_path}")

    frame_id = 0
    if frame_indices:
        max_index = sorted(frame_indices)[-1]

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            # No more frames. Recognition done.
            break

        # if frame_indices are listed
        if frame_indices is not None:
            # If all the frames in frame_indices have been processed
            # stop reading frames
            if frame_id > max_index:
                break
            if frame_id in frame_indices:
                yield (frame_id, frame)
        else:
            yield (frame_id, frame)
        frame_id += 1

    cap.release()


    

def read_vid_batch(video_path, frame_ids=None, batch_size=1):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    if frame_ids is not None:
        frame_ids = sorted(frame_ids)
    
    frame_id = 0
    frame_index = 0
    batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_ids is None or (frame_index < len(frame_ids) and frame_id == frame_ids[frame_index]):
            batch.append((frame_id, frame))
            if frame_ids is not None:
                frame_index += 1

        frame_id += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

        # Early termination if all specified frames are processed
        if frame_ids is not None and frame_index >= len(frame_ids):
            break

    if batch:
        yield batch

    cap.release()