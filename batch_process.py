#!/usr/bin/env python3
"""
Simple batch processing script using existing tools:
- generate_audio.py for audio generation
- generate_clap.py for CLAP scoring  
- visual_audio.py for spectrograms and waveforms

Takes a text file with sets of 3 prompts (separated by blank lines).
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from typing import List, Dict, Any


def parse_prompt_file(file_path: str) -> List[List[str]]:
    """Parse text file into groups of 3 prompts."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Remove empty lines and group into sets of 3
    non_empty_lines = [line for line in lines if line]
    
    prompt_sets = []
    for i in range(0, len(non_empty_lines), 3):
        if i + 2 < len(non_empty_lines):
            prompt_set = [non_empty_lines[i], non_empty_lines[i+1], non_empty_lines[i+2]]
            prompt_sets.append(prompt_set)
    
    print(f"Parsed {len(prompt_sets)} sets of prompts from {file_path}")
    return prompt_sets


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def get_clap_score(audio_file: str, text_prompt: str) -> float:
    """Get CLAP score using generate_clap.py."""
    cmd = ["python", "generate_clap.py", "--audio", audio_file, "--text", text_prompt]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Parse the score from output
        for line in result.stdout.strip().split('\n'):
            if "Score:" in line:
                return float(line.split("Score:")[-1].strip())
        return 0.0
    except:
        return 0.0


def process_set(prompts: List[str], set_idx: int, output_dir: str, **gen_args) -> Dict:
    """Process one set of prompts."""
    print(f"\n{'='*50}")
    print(f"Set {set_idx + 1}: {prompts}")
    print(f"{'='*50}")
    
    # Create set directory
    set_dir = os.path.join(output_dir, f"set_{set_idx + 1:03d}")
    os.makedirs(set_dir, exist_ok=True)
    
    # 1. Generate audio
    combined_audio = os.path.join(set_dir, "combined.wav")
    cmd = ["python", "generate_audio.py", "--prompts"] + prompts + ["--output", combined_audio]
    
    for key, val in gen_args.items():
        cmd.extend([f"--{key}", str(val)])
    
    if not run_command(cmd, "Audio generation"):
        return {"set_index": set_idx + 1, "success": False}
    
    # 2. Calculate CLAP score for combined audio with all prompts
    combined_prompt = " then ".join(prompts)  # Combine all prompts for CLAP evaluation
    clap_score = get_clap_score(combined_audio, combined_prompt)
    print(f"CLAP score: {clap_score:.4f}")
    
    # 3. Generate visualizations
    print(f"Creating visualizations...")
    
    # Waveform
    wave_png = os.path.join(set_dir, "waveform.png")
    run_command(["python", "visual_audio.py", combined_audio, "--waveform_png", wave_png], 
               "Waveform generation")
    
    # Spectrogram  
    spec_png = os.path.join(set_dir, "spectrogram.png")
    run_command(["python", "visual_audio.py", combined_audio, "--mel_png", spec_png],
               "Spectrogram generation")
    
    # Save results
    results = {
        "set_index": set_idx + 1, 
        "prompts": prompts, 
        "clap_score": clap_score, 
        "success": True
    }
    
    with open(os.path.join(set_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(set_dir, "prompts.txt"), 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i+1}. {prompt}\n")
        f.write(f"\nCLAP Score: {clap_score:.4f}\n")
    
    print(f"✅ Set {set_idx + 1} completed! CLAP: {clap_score:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process prompts using existing tools")
    parser.add_argument("--prompts_file", required=True, help="Text file with 3-line prompt sets")
    parser.add_argument("--output_dir", default="batch_results", help="Output directory")
    
    # Pass-through args for generate_audio.py
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--ttt_weights", default="ttt_finetune/ttt_module.pt") 
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Check required files
    required_files = ["generate_audio.py", "generate_clap.py", "visual_audio.py"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ {f} not found")
            sys.exit(1)
    
    # Parse prompts
    prompt_sets = parse_prompt_file(args.prompts_file)
    if not prompt_sets:
        print("❌ No valid prompt sets found!")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generation arguments
    gen_args = {k: v for k, v in vars(args).items() 
               if k in ["model_dir", "ttt_weights", "steps", "cfg_scale", "seed", "device"]}
    
    # Process all sets
    all_results = []
    successful = 0
    
    for i, prompts in enumerate(prompt_sets):
        result = process_set(prompts, i, args.output_dir, **gen_args)
        all_results.append(result)
        if result.get("success"):
            successful += 1
    
    # Summary
    successful_results = [r for r in all_results if r.get("success")]
    avg_clap = np.mean([r["clap_score"] for r in successful_results]) if successful_results else 0
    
    summary = {
        "total_sets": len(prompt_sets),
        "successful_sets": successful,
        "overall_average_clap": avg_clap,
        "results": all_results
    }
    
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(args.output_dir, "report.txt"), 'w') as f:
        f.write("BATCH PROCESSING REPORT\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Processed: {successful}/{len(prompt_sets)} sets\n")
        f.write(f"Overall avg CLAP: {avg_clap:.4f}\n\n")
        
        for r in successful_results:
            f.write(f"Set {r['set_index']}: {r['clap_score']:.4f}\n")
            for i, prompt in enumerate(r['prompts']):
                f.write(f"  {i+1}. {prompt}\n")
            f.write("\n")
    
    print(f"\n🎉 Completed! {successful}/{len(prompt_sets)} sets successful")
    if successful > 0:
        print(f"Overall average CLAP: {avg_clap:.4f}")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
