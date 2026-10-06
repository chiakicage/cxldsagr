"""Launch once, sample GPU process ownership, preserve the child exit status."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser(); p.add_argument('--output',type=pathlib.Path,required=True); p.add_argument('--selected-gpu',default='3'); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
if a.command and a.command[0]=='--': a.command=a.command[1:]
a.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,a.output/'driver.py')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ancestry(pid):
 rows=[]
 while pid>1:
  try:
   s=pathlib.Path(f'/proc/{pid}/stat').read_text(); tail=s[s.rfind(')')+2:].split(); parent=int(tail[1]); rows.append({'pid':pid,'ppid':parent,'start_ticks':int(tail[19])}); pid=parent
  except FileNotFoundError: break
 return rows
record={'command':a.command,'started_utc':now(),'driver_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
selected_uuid=subprocess.check_output(['nvidia-smi','--id='+a.selected_gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
rows=[]
with (a.output/'stdout.log').open('w') as out,(a.output/'stderr.log').open('w') as err,(a.output/'gpu_monitor.jsonl').open('w') as log:
 child=subprocess.Popen(a.command,stdout=out,stderr=err,start_new_session=True); record['pid']=child.pid
 while True:
  sample=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
  usage=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory','--format=csv,noheader'],capture_output=True,text=True)
  row={'gpu_utilization':usage.stdout,'gpu_utilization_returncode':usage.returncode,'time_utc':now(),'returncode':sample.returncode,'stdout':sample.stdout,'stderr':sample.stderr,'ancestry':{}}
  for line in sample.stdout.splitlines():
   if line.strip():
    pid=int(line.split(',')[0]); row['ancestry'][str(pid)]=ancestry(pid)
  rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
  code=child.poll()
  if code is not None: break
  time.sleep(2)
record.update(exit_code=code,completed_utc=now())
(a.output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
foreign=[]; observed=set()
exit_races=[]; previous={}
for row in rows:
 current={}
 for pid,chain in row['ancestry'].items():
  owned=int(pid)==child.pid or any(link['pid']==child.pid for link in chain)
  if owned:
   observed.add(int(pid)); current[pid]={'time_utc':row['time_utc'],'ancestry':chain}
  elif not chain and pid in previous:
   exit_races.append({'time_utc':row['time_utc'],'pid':int(pid),'prior_observation':previous[pid],'reason':'Previously confirmed child disappears from /proc between nvidia-smi and ancestry query; original samples retained.'})
  else: foreign.append({'time_utc':row['time_utc'],'pid':int(pid),'selected_gpu':any(line.startswith(pid+',') and selected_uuid in line for line in row['stdout'].splitlines())})
 previous=current
stamps=[datetime.datetime.fromisoformat(row['time_utc']).timestamp() for row in rows]
audit={'exit_code':code,'samples':len(rows),'observed_owned_pids':sorted(observed),'foreign_processes':foreign,'exit_races':exit_races,'monitor_errors':[row for row in rows if row['returncode'] or row['gpu_utilization_returncode']],'maximum_sample_gap_seconds':max((b-a for a,b in zip(stamps,stamps[1:])),default=0),'boundary':'Discrete observations of all GPUs; this does not exclude activity between samples or unrelated CPU work.'}
(a.output/'monitor_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit)); raise SystemExit(code if code else int(bool(any(row['selected_gpu'] for row in foreign) or audit['monitor_errors'])))
