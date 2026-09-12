"""Write a read-only comparison of three ablation jobs for cron monitoring."""

import json
from datetime import UTC, datetime
from pathlib import Path


def main():
    config=json.loads(Path('configs/experiments/dual_ablation_pilot.json').read_text())
    report={'time':datetime.now(UTC).isoformat(),'experiment':config['name'],'arms':{}}
    for arm in 'ABC':
        name=config['name']+'_'+arm
        logs=Path('artifacts/logs')/name
        path=logs/'status.json'
        if not path.exists():
            report['arms'][arm]={'status':'not_started'}
            continue
        status=json.loads(path.read_text())
        item={'status':status,'process_exists':Path('/proc',str(status['pid'])).exists()}
        path=logs/'training.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if row['kind'] in ['train','validation','reconstruction']:
                    item['last_'+row['kind']]=row
        path=Path('artifacts/reports')/name/'complete.json'
        if path.exists():item['result']=json.loads(path.read_text())
        report['arms'][arm]=item
    output=Path('artifacts/reports')/(config['name']+'_comparison.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix('.tmp')
    pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(output)
    print(json.dumps({arm:{'status':v['status'] if isinstance(v['status'],str) else v['status']['status']}
                      for arm,v in report['arms'].items()}))


if __name__=='__main__':main()
