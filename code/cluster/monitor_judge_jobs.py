#!/usr/bin/env python3
import subprocess
import json
import time
from pathlib import Path
import argparse
from collections import defaultdict


def get_job_status():
    try:
        result = subprocess.run(
            ['squeue', '-u', subprocess.check_output(['whoami']).decode().strip(), '--format=%i,%j,%t,%M,%N'],
            capture_output=True, text=True, check=True
        )
        
        jobs = []
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 5:
                    job_id, job_name, status, time_used, node = parts
                    jobs.append({
                        'job_id': job_id.strip(),
                        'job_name': job_name.strip(),
                        'status': status.strip(),
                        'time_used': time_used.strip(),
                        'node': node.strip()
                    })
        
        return jobs
    except subprocess.CalledProcessError:
        return []


def check_output_files(job_info_file, data_dir):
    if not Path(job_info_file).exists():
        return {}
    
    with open(job_info_file, 'r') as f:
        job_info = json.load(f)
    
    output_status = {}
    output_dir = Path(data_dir) / "judge_outputs"
    
    for job in job_info.get('jobs', []):
        output_file = output_dir / job['output_file']
        state_file = output_file.with_suffix('.state.json')
        
        status = {
            'output_exists': output_file.exists(),
            'state_exists': state_file.exists(),
            'output_size': output_file.stat().st_size if output_file.exists() else 0,
            'progress': None
        }
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    session_info = state.get('session_info', {})
                    if 'processed_count' in session_info and 'total_count' in session_info:
                        processed = session_info['processed_count']
                        total = session_info['total_count']
                        status['progress'] = f"{processed}/{total} ({processed/total*100:.1f}%)"
                        status['completed'] = session_info.get('completed', False)
            except:
                pass
        
        output_status[job['job_id']] = status
    
    return output_status


def print_status_summary(jobs, output_status, job_info_file):
    judge_jobs = [j for j in jobs if 'judge_' in j['job_name']]
    
    for job in judge_jobs:
        progress = output_status.get(job['job_id'], {}).get('progress', '')
        print(f"{job['job_id']} {job['job_name']:<25} {job['status']:<8} {progress}")
    
    if output_status:
        completed = sum(1 for s in output_status.values() if s.get('completed', False))
        existing = sum(1 for s in output_status.values() if s['output_exists'])
        print(f"\nCompleted: {completed}/{len(output_status)}, Existing: {existing}/{len(output_status)}")


def main():
    parser = argparse.ArgumentParser(description="Monitor judge collection jobs")
    parser.add_argument('--job-info', type=str, default='submitted_jobs.json')
    parser.add_argument('--data-dir', type=str, default='/scratch/$USER/faviscore/data')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    
    data_dir = args.data_dir.replace('$USER', subprocess.check_output(['whoami']).decode().strip())
    
    try:
        while True:
            if args.watch:
                subprocess.run(['clear'])
            
            jobs = get_job_status()
            output_status = check_output_files(args.job_info, data_dir)
            print_status_summary(jobs, output_status, args.job_info)
            
            if not args.watch:
                break
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
