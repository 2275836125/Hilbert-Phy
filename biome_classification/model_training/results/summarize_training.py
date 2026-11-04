"""Consolidate all formal internal five-fold test results. Run from any directory."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'
MODELS = ['hilbert_resnet18','hilbert_resnet34','hilbert_resnet50','rowwise_resnet18','rowwise_resnet34','rowwise_resnet50','1d_resnet18','deepphylo_taxonomy','popphy_cnn','random_forest_rowwise','random_forest_random','logistic_regression_rowwise','logistic_regression_random']
KEYS = ['accuracy','macro_f1','macro_auroc']

def read_json(p):
    return json.loads(p.read_text())

def location(model, level):
    if model in ('deepphylo_taxonomy','popphy_cnn'):
        return ROOT/'phylogeny_baselines'/model/'results/formal'/f'level_{level}', None
    for family in ('random_forest','logistic_regression'):
        if model.startswith(family+'_'):
            return ROOT/'classical_order_control/results'/family/f'level_{level}', model[len(family)+1:]
    return OUT/'experiments'/f'level_{level}'/model, None

def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

folds, groups, statuses = [], [], []
for level in range(1,6):
    for model in MODELS:
        parent, order = location(model, level)
        local, ids_seen = [], set()
        pooled_summary = read_json(parent/'summary.json') if (parent/'summary.json').exists() else None
        if order and pooled_summary:
            pooled_summary = pooled_summary['orders'][order]
        for fold in range(1,6):
            folder = parent/f'fold_{fold}'
            if order:
                folder /= order
            path = folder/'result.json'
            state = dict(model=model,level=level,fold=fold,status='missing',source_result=str(path.relative_to(ROOT)))
            if not path.exists():
                statuses.append(state)
                continue
            result = read_json(path)
            assert not result.get('config',{}).get('smoke',False), path
            assert result.get('status','complete') == 'complete', path
            metric = result.get('metrics',result)
            if order:
                with (folder/'test_predictions.csv').open(newline='') as f:
                    reader=csv.reader(f); header=next(reader)
                    ii,yi,pi = [header.index(k) for k in ['sample_id','true_index','predicted_index']]
                    records=[(r[ii],int(r[yi]),int(r[pi])) for r in reader]
                ids=[r[0] for r in records]
                y=np.array([r[1] for r in records]); pred=np.array([r[2] for r in records])
            else:
                with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                    ids=data['ids'].astype(str).tolist(); y=data['y']; pred=data['probabilities'].argmax(1)
            assert len(ids)==len(set(ids)) and not ids_seen.intersection(ids), folder
            ids_seen.update(ids)
            cache=ROOT/'shared_cache'/f'level_{level}'/f'fold_{fold}'
            expected=np.load(cache/'test_ids.npy',allow_pickle=False).astype(str)
            expected_y=np.load(cache/'test_y.npy',allow_pickle=False)
            assert ids==expected.tolist() and np.array_equal(y,expected_y), folder
            classes=len(read_json(cache/'classes.json'))
            cm=np.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes)
            den=cm.sum(0)+cm.sum(1)
            f1=np.divide(2*np.diag(cm),den,out=np.zeros(classes,dtype=float),where=den!=0).mean()
            assert abs(f1-metric['macro_f1']) < 1e-10, folder
            row=dict(model=model,level=level,fold=fold,test_samples=len(y),training_classes=classes,
                     accuracy=float(np.mean(y==pred)),macro_f1=metric['macro_f1'],macro_auroc=metric['macro_auroc'],
                     best_epoch=result.get('best_epoch',''),best_validation_loss=result.get('best_validation_loss',''),
                     parameters=result.get('parameters',''),source_result=str(path.relative_to(ROOT)))
            folds.append(row);local.append(row)
            state['status']='complete';statuses.append(state)
        g=dict(model=model,level=level,completed_folds=len(local),expected_folds=5,test_samples_total=sum(r['test_samples'] for r in local),training_classes=local[0]['training_classes'] if local else '')
        for key in KEYS:
            values=[r[key] for r in local]
            g[key+'_mean']=mean(values) if values else None
            g[key+'_std']=stdev(values) if len(values)>1 else None
        g['pooled_oof_accuracy']=sum(r['accuracy']*r['test_samples'] for r in local)/g['test_samples_total'] if local else None
        for key in ['macro_f1','macro_auroc']:
            g['pooled_oof_'+key]=pooled_summary['pooled_oof'][key] if pooled_summary and len(local)==5 else None
            if pooled_summary and len(local)==5:
                for stat in ['mean','std']:
                    assert abs(g[key+'_'+stat]-pooled_summary['metrics'][key][stat])<1e-10, (model,level,key,stat)
        groups.append(g)
        print(f'Level {level} {model}: {len(local)}/5',flush=True)
assert len(folds)==325 and all(g['completed_folds']==5 for g in groups)
write_csv('training_fold_metrics.csv',folds)
write_csv('training_method_level_summary.csv',groups)
write_csv('training_task_status.csv',statuses)
(OUT/'training_summary.json').write_text(json.dumps(dict(evaluation='internal held-out test folds, not training-set fitting metrics',standard_deviation_ddof=1,groups=groups),ensure_ascii=False,indent=2)+'\n')
lines=['# 训练实验内部五折测试结果汇总','','全部13种配置统一汇总：Hilbert/Rowwise ResNet18、34、50，1D ResNet18，DeepPhylo、PopPhyCNN，以及随机森林/逻辑回归各含rowwise、random两种顺序。共5层×13配置×5折=325项，全部完成。','','## 指标口径','','这里的“训练实验指标”来自各折留出测试集，不是训练集拟合指标，也不是用于早停的验证集指标。神经网络恢复验证损失最佳的模型后评价测试集。未重新训练或选择模型。','','表中为五折指标均值 ± 样本标准差（ddof=1），以百分数表示，标准差单位为百分点；不是置信区间。各折测试样本互斥，另在CSV/JSON中单列pooled OOF指标，不能与五折均值混用。','','Macro-F1平均所有训练类别；Macro-AUROC为原训练协议的逐类one-vs-rest宏平均。Accuracy由已有测试预测补算。已核对每折预测ID、标签与共享测试缓存一致，折间ID互斥，重算Macro-F1与原结果一致，汇总均值及标准差与各方法原summary.json一致。AUROC及pooled OOF F1/AUROC引用已有结果，未重新计算概率排序指标。','','## 各层评价样本','','| 层级 | OOF样本总数 | 类别数 | 完成配置 |','|---|---:|---:|---:|']
for level in range(1,6):
    gg=[g for g in groups if g['level']==level]
    assert len({g['test_samples_total'] for g in gg})==1
    lines.append(f"| {level} | {gg[0]['test_samples_total']} | {gg[0]['training_classes']} | 13/13 |")
for level in range(1,6):
    lines += ['',f'## Level {level}','','按Macro-F1均值降序，全部方法置于同一张表。','','| 方法 | Accuracy | Macro-F1 | Macro-AUROC |','|---|---:|---:|---:|']
    for g in sorted([g for g in groups if g['level']==level],key=lambda g:g['macro_f1_mean'],reverse=True):
        lines.append('| '+g['model']+' | '+' | '.join(f"{100*g[k+'_mean']:.2f} ± {100*g[k+'_std']:.2f}" for k in KEYS)+' |')
lines += ['', '## 文件与来源','','- `training_method_level_summary.csv`：65行方法×层级汇总，含五折均值、标准差和pooled OOF指标。','- `training_fold_metrics.csv`：325行逐折指标，含最佳epoch、验证损失、参数量（原结果提供时）和来源路径。','- `training_task_status.csv`：325项任务状态与来源。','- `training_summary.json`：相同汇总的机器可读版本。','- `summarize_training.py`：统一汇总脚本。','','CSV/JSON指标保留0–1尺度，报告以百分数展示。原始模型产物保留在原目录；所有新增汇总文件统一保存于`model_training/results/`。','','原始结果来源：`results/experiments/`、`phylogeny_baselines/{deepphylo_taxonomy,popphy_cnn}/results/formal/`、`classical_order_control/results/`。排除smoke、verification和检查任务。','','固定4096属特征和min20样本为全数据预筛选，不能解释为完全独立的训练折内特征筛选。各层样本和类别集合不同，与外部验证对比时也需注意评价集合差异。','']
(OUT/'training_results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'Wrote {len(groups)} groups and {len(folds)} folds to {OUT}',flush=True)
