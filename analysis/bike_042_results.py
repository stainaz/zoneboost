import json, os, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
import zoneboost
from zoneboost import ZoneBoostRegressor, ZoneBoostClassifier, ConformalizedQuantileRegressor, BootstrapStability, ZoneForest, ZoneBoostTimeSeries, ZoneProfileEncoder, DepthTransformer, CategoricalDepthTransformer, DepthCrowd, ConditionalZoneGrid, ZoneFeatureSpace, compare_models, flag_drift, compile_to_sql

OUT=Path('analysis_output'); OUT.mkdir(exist_ok=True)
status={}; metrics=[]
def ok(name, data=None): status[name]={'status':'ok','data':data}
def fail(name,e): status[name]={'status':'failed','error':f'{type(e).__name__}: {e}'}
def reg_metric(name,y,p):
    d={'model':name,'RMSE':float(mean_squared_error(y,p)**0.5),'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}; metrics.append(d); return d

bike=fetch_ucirepo(id=275)
df=bike.data.features.copy(); targ=bike.data.targets.copy()
if 'cnt' in targ.columns: df['cnt']=targ['cnt'].to_numpy()
elif 'cnt' not in df: raise RuntimeError('cnt unavailable')
df['dteday']=pd.to_datetime(df['dteday']); df=df.sort_values(['dteday','hr']).reset_index(drop=True)
df=df.drop(columns=[c for c in ['instant','casual','registered'] if c in df])
cat=[c for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit'] if c in df]
num=[c for c in ['temp','atemp','hum','windspeed'] if c in df]
features=cat+num
for c in cat: df[c]=df[c].astype('category')
cut=int(len(df)*.8); X=df[features]; y=df.cnt.astype(float)
Xtr,Xte=X.iloc[:cut].copy(),X.iloc[cut:].copy(); ytr,yte=y.iloc[:cut],y.iloc[cut:]
summary={'zoneboost_version':zoneboost.__version__,'rows':len(df),'features':features,'train_rows':len(Xtr),'test_rows':len(Xte),'train_period':[str(df.dteday.iloc[0].date()),str(df.dteday.iloc[cut-1].date())],'test_period':[str(df.dteday.iloc[cut].date()),str(df.dteday.iloc[-1].date())]}

# Basic graphs
pd.DataFrame({'date':df.dteday,'cnt':y}).groupby('date').cnt.sum().plot(figsize=(12,4),title='Daily bike rentals'); plt.ylabel('rentals'); plt.tight_layout(); plt.savefig(OUT/'daily_rentals.png'); plt.close()
df.groupby('hr',observed=True).cnt.mean().plot(kind='bar',figsize=(10,4),title='Average rentals by hour'); plt.ylabel('average rentals'); plt.tight_layout(); plt.savefig(OUT/'rentals_by_hour.png'); plt.close()

base_params=dict(n_rounds=100,learning_rate=.05,max_zones=7,max_pair_interactions=10,categorical_features=cat,validation_fraction=.2,n_iter_no_change=15,random_state=42)
try:
    zb=ZoneBoostRegressor(**base_params,spline_zones=num,spline_shrinkage_m=10.0,track_reliability=True).fit(Xtr,ytr)
    p=zb.predict(Xte); ok('ZoneBoostRegressor',reg_metric('ZoneBoost spline',yte,p)); pd.DataFrame({'actual':yte.to_numpy(),'prediction':p,'residual':yte.to_numpy()-p}).to_csv(OUT/'zoneboost_predictions.csv',index=False)
    plt.figure(figsize=(6,6)); plt.scatter(yte,p,s=7,alpha=.2); lo=min(yte.min(),p.min()); hi=max(yte.max(),p.max()); plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('actual'); plt.ylabel('predicted'); plt.title('ZoneBoost actual vs predicted'); plt.tight_layout(); plt.savefig(OUT/'actual_vs_predicted.png'); plt.close()
    imp=zb.feature_importance(Xte); pd.DataFrame(imp).to_csv(OUT/'feature_importance.csv',index=False)
    exp=zb.explain(Xte.iloc[:20]); exp.to_csv(OUT/'local_explanations.csv',index=False)
except Exception as e: fail('ZoneBoostRegressor',e); zb=None

# Ridge baseline
try:
    pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
    ridge=Pipeline([('pre',pre),('model',Ridge(alpha=10.0))]).fit(Xtr,ytr); pr=ridge.predict(Xte); ok('RidgeBaseline',reg_metric('Ridge raw',yte,pr))
except Exception as e: fail('RidgeBaseline',e)

# ZoneForest
try:
    est=ZoneBoostRegressor(**base_params)
    zf=ZoneForest(estimator=est,n_estimators=8,row_subsample=.8,col_subsample=.8,n_jobs=-1,random_state=42).fit(Xtr,ytr)
    ok('ZoneForest',reg_metric('ZoneForest',yte,zf.predict(Xte)))
except Exception as e: fail('ZoneForest',e)

# CQR
try:
    cqr=ConformalizedQuantileRegressor(alpha=.1,categorical_features=cat,random_state=42).fit(Xtr,ytr)
    lo,hi=cqr.predict_interval(Xte); cov=float(np.mean((yte.to_numpy()>=lo)&(yte.to_numpy()<=hi))); wd=hi-lo
    ok('ConformalizedQuantileRegressor',{'coverage':cov,'mean_width':float(np.mean(wd)),'median_width':float(np.median(wd))})
    pd.DataFrame({'actual':yte.to_numpy(),'lower':lo,'upper':hi}).to_csv(OUT/'conformal_intervals.csv',index=False)
except Exception as e: fail('ConformalizedQuantileRegressor',e)

# Profile, depth, crowd, conditional grid, feature space
try:
    prof=ZoneProfileEncoder(columns=features,categorical_features=cat,group_name='profile'); ptr=prof.fit_transform(Xtr,ytr); pte=prof.transform(Xte); ok('ZoneProfileEncoder',{'train_shape':list(ptr.shape),'test_shape':list(pte.shape)})
except Exception as e: fail('ZoneProfileEncoder',e); ptr=pte=None
try:
    dt=DepthTransformer(columns=num,group_name='weather'); dtr=dt.fit_transform(Xtr[num]); dte=dt.transform(Xte[num]); ok('DepthTransformer',{'train_shape':list(dtr.shape)})
    cd=CategoricalDepthTransformer(columns=['season','mnth','hr','weekday','workingday'],group_name='calendar'); ctr=cd.fit_transform(Xtr[cat]); cte=cd.transform(Xte[cat]); ok('CategoricalDepthTransformer',{'train_shape':list(ctr.shape)})
    experts_tr=pd.concat([dtr.filter(like='__coreness'),ctr.filter(like='__coreness')],axis=1); experts_te=pd.concat([dte.filter(like='__coreness'),cte.filter(like='__coreness')],axis=1)
    crowd=DepthCrowd(columns=list(experts_tr.columns),rank_normalize=True,vote_threshold=.05,group_name='crowd'); crtr=crowd.fit_transform(experts_tr); crte=crowd.transform(experts_te); ok('DepthCrowd',{'train_shape':list(crtr.shape),'numeric_columns':list(crtr.select_dtypes(include=np.number).columns)})
except Exception as e: fail('DepthPipeline',e)
try:
    cg=ConditionalZoneGrid(segment_columns=['workingday'],grid_columns=['temp','hum'],group_name='workday_weather'); cgtr=cg.fit_transform(Xtr,ytr); cgte=cg.transform(Xte); ok('ConditionalZoneGrid',{'train_shape':list(cgtr.shape)})
except Exception as e: fail('ConditionalZoneGrid',e)
try:
    zfs=ZoneFeatureSpace(profile_columns=features,categorical_features=cat,depth_groups={'weather':num},categorical_depth_groups={'calendar':['season','mnth','hr','weekday']},conditional_grids=[{'segment_columns':['workingday'],'grid_columns':['temp','hum'],'group_name':'workday_weather'}]); ftr=zfs.fit_transform(Xtr,ytr); fte=zfs.transform(Xte); ok('ZoneFeatureSpace',{'train_shape':list(ftr.shape),'test_shape':list(fte.shape)})
except Exception as e: fail('ZoneFeatureSpace',e)

# Classification high demand
try:
    threshold=float(ytr.quantile(.75)); yc_tr=(ytr>=threshold).astype(int); yc_te=(yte>=threshold).astype(int)
    clf=ZoneBoostClassifier(n_rounds=100,learning_rate=.05,categorical_features=cat,max_pair_interactions=10,validation_fraction=.2,n_iter_no_change=15,calibrate=True,calibration_fraction=.1,refit_on_full_data=True,random_state=42).fit(Xtr,yc_tr)
    prob=clf.predict_proba(Xte)[:,1]; pred=clf.predict(Xte)
    cm=confusion_matrix(yc_te,pred); pd.DataFrame(cm).to_csv(OUT/'classification_confusion_matrix.csv',index=False)
    ok('ZoneBoostClassifier',{'threshold':threshold,'accuracy':float(accuracy_score(yc_te,pred)),'balanced_accuracy':float(balanced_accuracy_score(yc_te,pred)),'macro_F1':float(f1_score(yc_te,pred,average='macro')),'ROC_AUC':float(roc_auc_score(yc_te,prob)),'log_loss':float(log_loss(yc_te,np.c_[1-prob,prob]))})
except Exception as e: fail('ZoneBoostClassifier',e)

# Drift / time series
try:
    m11=(df.yr.astype(int)==0); m12=(df.yr.astype(int)==1); old=ZoneBoostRegressor(**base_params).fit(df.loc[m11,features],y[m11]); new=ZoneBoostRegressor(**base_params).fit(df.loc[m12,features],y[m12]); drift=compare_models(old,new,df.loc[m12,features],y[m12]);
    if isinstance(drift,dict): json.dump(drift,open(OUT/'drift.json','w'),default=str,indent=2)
    else: pd.DataFrame(drift).to_csv(OUT/'drift.csv',index=False)
    ok('compare_models',{'type':type(drift).__name__})
    try: ok('flag_drift',flag_drift(old,new,df.loc[m12,features],y[m12],alpha=.1))
    except Exception as ee: fail('flag_drift',ee)
except Exception as e: fail('Drift',e)
try:
    tsbase=ZoneBoostRegressor(n_rounds=40,learning_rate=.05,categorical_features=cat,max_pair_interactions=5,random_state=42)
    ts=ZoneBoostTimeSeries(estimator=tsbase,time_col='dteday',freq='Q',window='expanding',min_train_periods=2).fit(df[['dteday']+features],y)
    sr=ts.stability_report(); pd.DataFrame(sr).to_csv(OUT/'time_series_stability.csv',index=False); ok('ZoneBoostTimeSeries',{'report_shape':list(pd.DataFrame(sr).shape)})
except Exception as e: fail('ZoneBoostTimeSeries',e)

# Bootstrap limited for runtime
try:
    bs=BootstrapStability(ZoneBoostRegressor(n_rounds=40,learning_rate=.05,categorical_features=cat,max_pair_interactions=5,random_state=42),n_bootstrap=5,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    inc=bs.inclusion_frequency(); pd.DataFrame(inc).to_csv(OUT/'bootstrap_inclusion.csv',index=False); ok('BootstrapStability',{'rows':len(pd.DataFrame(inc))})
except Exception as e: fail('BootstrapStability',e)

# SQL export
try:
    sm=ZoneBoostRegressor(n_rounds=20,learning_rate=.05,categorical_features=cat,max_interaction_order=2,max_pair_interactions=5,random_state=42).fit(Xtr,ytr); sql=compile_to_sql(sm,table_name='bike_scoring'); open(OUT/'model.sql','w').write(sql); ok('compile_to_sql',{'characters':len(sql)})
except Exception as e: fail('compile_to_sql',e)

pd.DataFrame(metrics).sort_values('RMSE').to_csv(OUT/'model_metrics.csv',index=False)
summary['metrics']=metrics; summary['functionality']=status
json.dump(summary,open(OUT/'summary.json','w'),indent=2,default=str)
print(json.dumps(summary,indent=2,default=str))
