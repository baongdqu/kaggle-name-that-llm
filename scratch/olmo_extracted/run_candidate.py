"""Baseline sees both T4s; it exits before the owned single-GPU process."""
import hashlib,importlib.metadata,json,os,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'candidate-config.json').read_text())
assert os.name!='nt' and Path('/kaggle/input').is_dir() and C['source_ready']
mode=os.environ.get('NTL_RUN_MODE','candidate')
assert mode in ('candidate','probe')
assert C['cloud_probe_authorized'] if mode=='probe' else C['candidate_admitted']
for name,digest in C['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
S=json.loads((HERE/'olmo-spec.json').read_text())
assert {name:importlib.metadata.version(name) for name in S['core_versions']}==S['core_versions']
mapping=[]
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],text=True).strip().splitlines():
    index,uuid,name=[x.strip() for x in line.split(',',2)]
    mapping.append(dict(physical_index=int(index),uuid=uuid,name=name))
mapping.sort(key=lambda row:row['physical_index'])
assert [row['physical_index'] for row in mapping]==[0,1] and all('T4' in row['name'] for row in mapping)
Path('/kaggle/temp').mkdir(exist_ok=True)
cache=Path(tempfile.mkdtemp(prefix='ntl-two-stage-olmo-',dir='/kaggle/temp'))
env=os.environ.copy()
env['NTL_CANDIDATE_TEMP']=str(cache)
env['HF_HUB_OFFLINE']=env['TRANSFORMERS_OFFLINE']='1'
assert not env.get('NTL_QUERY_CSV') and not env.get('NTL_SMOKE')
if mode=='probe':
    import pandas as pd
    count=C['probe_rows']
    assert count in (3,5)
    texts=S['synthetic_texts']+['A small wooden bridge crosses the quiet stream beside the old mill.',
        'The experiment compares a fixed control with a single additional measurement.']
    ids=['synthetic-a','synthetic-b','synthetic-c','synthetic-d','synthetic-e']
    query=cache/'synthetic-query.csv'
    pd.DataFrame({'id':ids[:count],'text':texts[:count]}).to_csv(query,index=False)
    env['NTL_QUERY_CSV']=str(query)
    Path('/kaggle/working/synthetic_probe_inputs.json').write_text(json.dumps(dict(ids=ids[:count],texts=texts[:count]),indent=2))
deadline=time.monotonic()+(1800 if mode=='probe' else 10800)
stages=[('baseline','run_baseline.py',','.join(row['uuid'] for row in mapping)),
        ('olmo','compute_olmo.py',mapping[0]['uuid'])]
reports=[]
for name,script,devices in stages:
    child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=devices
    start=time.monotonic()
    samples=both_active=0
    with Path('/kaggle/working',name+'.log').open('w',buffering=1) as handle:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/script)],env=child_env,cwd='/kaggle/working',
            stdout=handle,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                assert time.monotonic()<deadline,'Bounded whole-run timeout; no retry'
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],text=True,timeout=10)
                processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                    '--format=csv,noheader,nounits'],text=True,timeout=10)
                observed=[]
                for line in gpu.strip().splitlines():
                    index,uuid,util,memory=[x.strip() for x in line.split(',')]
                    observed.append(dict(physical_index=int(index),uuid=uuid,utilization=int(util),memory_MiB=int(memory)))
                observed.sort(key=lambda row:row['physical_index'])
                assert [row['uuid'] for row in observed]==[row['uuid'] for row in mapping]
                active=all(row['utilization']>0 and row['memory_MiB']>0 for row in observed)
                samples+=1;both_active+=int(active)
                with Path('/kaggle/working/gpu_stage_telemetry.jsonl').open('a') as telemetry:
                    telemetry.write(json.dumps(dict(stage=name,PID=child.pid,elapsed=time.monotonic()-start,
                        physical_devices=observed,compute_processes=processes.strip().splitlines(),
                        both_physical_GPUs_active=active))+'\n')
                time.sleep(1)
            assert child.returncode==0,(name,'failed; no automatic retry',child.returncode)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    reports.append(dict(stage=name,PID=child.pid,CUDA_VISIBLE_DEVICES=devices,seconds=time.monotonic()-start,
        exit_code=0,telemetry_samples=samples,both_physical_GPUs_active_samples=both_active))
    # The child has exited; no baseline CUDA model survives in this orchestrator.
    assert name!='baseline' or (cache/'baseline-state.npz').is_file()
assert time.monotonic()<deadline
subprocess.run([sys.executable,'-u',str(HERE/'finish_prediction.py')],env=env,cwd='/kaggle/working',check=True,
    timeout=min(120,max(1,deadline-time.monotonic())))
assert json.loads(Path('/kaggle/working/candidate_audit.json').read_text())['status']=='TWO_STAGE_GAUSSIAN_PLUS_OWN_OLMO03_COMPLETE'
Path('/kaggle/working/sequential_execution_audit.json').write_text(json.dumps(dict(mode=mode,stages=reports,
    no_parallel_baseline_and_OLMo=True,baseline_visible_physical_GPUs=[0,1],OLMo_visible_physical_GPU=0,
    parent_model_initializations=0,automatic_retries=0),indent=2))
