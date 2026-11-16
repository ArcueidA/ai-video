"""
Wan 2.2 14B (LightX2V + Dancing LoRA) + Practical-RIFE 4x (60fps) 10s Video Pipeline
Intended to run inside a Google Colab runtime after ComfyUI & Practical-RIFE setup.
"""

import glob
import os
import subprocess
import torch


def run_720p_60fps_pipeline(
    image_path: str = "/content/ComfyUI/input/character.jpg",
    output_path: str = "/content/ComfyUI/output/wan22_dancing_720p_60fps_10s.mp4",
    seed: int = 815400480,
):
    positive_prompt = (
        "A beautiful young woman with long dark wavy hair wearing a beige knitted cardigan "
        "over a white top, dancing gracefully and energetically to upbeat music on a sunny "
        "tree-lined city street, rhythmic body sway, expressive hand gestures, natural hair "
        "movement, warm smile, smooth cinematic motion, high quality. b3ll13-d8nc3r."
    )
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走"
    )

    os.chdir("/content/ComfyUI")
    generated_ok = False
    for try_frames in (101, 81):
        try:
            print(f"Generating 720p (720x1280) with frames={try_frames}, steps=8...")
            generate_video(  # noqa: F821 (defined in Colab notebook setup cell)
                image_path=image_path,
                LoRA_Strength=1.0,
                rel_l1_thresh=0,
                start_percent=0.2,
                end_percent=1.0,
                positive_prompt=positive_prompt,
                prompt_assist="b3ll13-d8nc3r",
                negative_prompt=negative_prompt,
                width=720,
                height=1280,
                seed=seed,
                steps=8,
                cfg_scale=1,
                sampler_name="euler",
                scheduler="simple",
                frames=try_frames,
                fps=16,
                output_format="mp4",
                overwrite=True,
                use_lora=False,
                use_lora2=False,
                LoRA_Strength2=1.0,
                use_lora3=False,
                LoRA_Strength3=1.0,
                use_lightx2v=True,
                lightx2v_Strength=3.0,
                lightx2v_steps=8,
                use_pusa=True,
                pusa_Strength=1.5,
                pusa_steps=8,
                use_sage_attention=False,
                enable_flow_shift=True,
                shift=8,
                enable_flow_shift2=True,
                shift2=8,
                end_step1=4,
            )
            generated_ok = True
            break
        except Exception as exc:
            print(f"frames={try_frames} failed ({exc}), clearing memory and trying fallback...")
            try:
                clear_memory()  # noqa: F821
            except Exception:
                pass
            torch.cuda.empty_cache()

    if not generated_ok:
        raise RuntimeError("720p generation failed for all frame counts.")

    clear_memory()  # noqa: F821

    video_folder = "/content/ComfyUI/output/"
    raw_videos = sorted(glob.glob(os.path.join(video_folder, "*.mp4")), key=os.path.getctime)
    raw_mp4 = raw_videos[-1]

    os.chdir("/content/Practical-RIFE")
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["FFMPEG_LOGLEVEL"] = "quiet"
    subprocess.run(
        ["python3", "inference_video.py", "--multi=4", "--fps=60", f"--video={raw_mp4}", "--scale=1"],
        check=True,
    )

    interp_mp4 = sorted(glob.glob(os.path.join(video_folder, "*.mp4")), key=os.path.getctime)[-1]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            interp_mp4,
            "-filter_complex",
            "[0:v]reverse,trim=start_frame=2[r];[0:v][r]concat=n=2:v=1:a=0,fps=60,trim=duration=10.0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-preset",
            "fast",
            output_path,
            "-loglevel",
            "error",
        ],
        check=True,
    )
    print(f"Saved 720p 60fps 10s video to: {output_path}")


if __name__ == "__main__":
    run_720p_60fps_pipeline()
