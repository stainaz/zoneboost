import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def ok(name,data=None): status[name]={'status':'ok','data':data}
def fail(name,e): status[name]={'status':'failed','error':f'{type(e).__name__}: {e}'}
def reg(name,y,p):
 d={'model':name,'RMSE':float(mean_squared_error(y,p)**0.5),'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}; metrics.append(d); return d

assert zoneboost.__version__=='0.42.0', zoneboost.__version__
bike=fetch_ucirepo(id=275); df=bike.data.features.copy(); t=bike.data.targets.copy()
if 'cnt' in t: df['cnt']=t['cnt'].to_numpy()
df['dteday']=pd.to_datetime(df['dteday']); df=df.sort_values(['dteday','hr']).reset_index(drop=True)
df=df.drop(columns=[c for c in ['instant','casual','registered'] if c in df])
cat=[c for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit'] if c in df]
num=[c for c in ['temp','atemp','hum','windspeed'] if c in df]; features=cat+num
for c in cat: df[c]=df[c].astype('category')
X=df[features]; y=df.cnt.astype(float); cut=int(len(df)*.8)
Xtr,Xte=X.iloc[:cut].copy(),X.iloc[cut:].copy(); ytr,yte=y.iloc[:cut],y.iloc[cut:]
summary={'zoneboost_version':zoneboost.__version__,'rows':len(df),'train_rows':len(Xtr),'test_rows':len(Xte),'features':features,'train_period':[str(df.dteday.iloc[0].date()),str(df.dteday.iloc[cut-1].date())],'test_period':[str(df.dteday.iloc[cut].date()),str(df.dteday.iloc[-1].date())]}

# EDA
pd.DataFrame({'date':df.dteday,'cnt':y}).groupby('date').cnt.sum().plot(figsize=(12,4),title='Daily bike rentals'); plt.ylabel('Rentals'); plt.tight_layout(); plt.savefig(OUT/'daily_rentals.png'); plt.close()
df.groupby('hr',observed=True).cnt.mean().plot(kind='bar',figsize=(10,4),title='Average rentals by hour'); plt.ylabel('Average rentals'); plt.tight_layout(); plt.savefig(OUT/'rentals_by_hour.png'); plt.close()

base=dict(n_rounds=80,learning_rate=.05,max_zones=7,max_pair_interactions=10,categorical_features=cat,validation_fraction=.2,n_iter_no_change=12,random_state=42)
try:
 zb=ZoneBoostRegressor(**base,spline_zones=num,spline_shrinkage_m=10.0,track_reliability=True).fit(Xtr,ytr); p=zb.predict(Xte); ok('ZoneBoostRegressor',reg('ZoneBoost spline',yte,p))
 pd.DataFrame({'actual':yte.to_numpy(),'prediction':p,'residual':yte.to_numpy()-p}).to_csv(OUT/'zoneboost_predictions.csv',index=False)
 plt.figure(figsize=(6,6)); plt.scatter(yte,p,s=7,alpha=.2); lo=min(yte.min(),p.min()); hi=max(yte.max(),p.max()); plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('ZoneBoost actual vs predicted'); plt.tight_layout(); plt.savefig(OUT/'actual_vs_predicted.png'); plt.close()
 plt.figure(figsize=(9,4)); plt.scatter(p,yte.to_numpy()-p,s=7,alpha=.2); plt.axhline(0,ls='--'); plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.title('Residuals'); plt.tight_layout(); plt.savefig(OUT/'residuals.png'); plt.close()
 imp=zb.feature_importance(Xte); imp_s=imp if isinstance(imp,pd.Series) else pd.Series(imp); imp_s.sort_values(ascending=False).to_csv(OUT/'feature_importance.csv',header=['importance'])
 imp_s.sort_values().tail(15).plot(kind='barh',figsize=(9,6),title='Top ZoneBoost terms'); plt.tight_layout(); plt.savefig(OUT/'feature_importance.png'); plt.close()
 zb.explain(Xte.iloc[:20]).to_csv(OUT/'local_explanations.csv',index=False)
 try:
  from zoneboost.eda import prediction_waterfall, signed_contribution_profile, zone_boxplot
  prediction_waterfall(zb,Xte.iloc[[0]],row=0); plt.tight_layout(); plt.savefig(OUT/'prediction_waterfall.png'); plt.close()
  signed_contribution_profile(zb,Xte,feature='temp'); plt.tight_layout(); plt.savefig(OUT/'signed_temp_profile.png'); plt.close()
  zone_boxplot(Xtr,ytr,column='temp'); plt.tight_layout(); plt.savefig(OUT/'temp_zone_boxplot.png'); plt.close(); ok('zoneboost.eda')
 except Exception as e: fail('zoneboost.eda',e)
except Exception as e: fail('ZoneBoostRegressor',e); zb=None

try:
 pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)]); model=Pipeline([('pre',pre),('ridge',Ridge(alpha=10))]).fit(Xtr,ytr); ok('RidgeBaseline',reg('Ridge raw',yte,model.predict(Xte)))
except Exception as e: fail('RidgeBaseline',e)

try:
 zf=ZoneForest(ZoneBoostRegressor(**base),n_estimators=6,max_samples=.8,max_features=.8,n_jobs=2,random_state=42).fit(Xtr,ytr); ok('ZoneForest',reg('ZoneForest',yte,zf.predict(Xte)))
except Exception as e: fail('ZoneForest',e)

try:
 template=ZoneBoostRegressor(n_rounds=60,learning_rate=.05,categorical_features=cat,max_pair_interactions=8,random_state=42)
 cqr=ConformalizedQuantileRegressor(estimator=template,alpha=.1,calibration_fraction=.2,random_state=42).fit(Xtr,ytr); lo,hi=cqr.predict_interval(Xte); width=hi-lo; cov=float(np.mean((yte.to_numpy()>=lo)&(yte.to_numpy()<=hi))); ok('ConformalizedQuantileRegressor',{'coverage':cov,'mean_width':float(width.mean()),'median_width':float(np.median(width))}); pd.DataFrame({'actual':yte.to_numpy(),'lower':lo,'upper':hi}).to_csv(OUT/'conformal_intervals.csv',index=False)
 plt.figure(figsize=(12,4)); n=250; x=np.arange(n); plt.fill_between(x,lo[:n],hi[:n],alpha=.25); plt.plot(x,yte.iloc[:n].to_numpy(),lw=1); plt.title('90% conformal intervals'); plt.tight_layout(); plt.savefig(OUT/'conformal_intervals.png'); plt.close()
except Exception as e: fail('ConformalizedQuantileRegressor',e)

try:
 prof=ZoneProfileEncoder(max_zones=7,min_zone_abs=20).fit(Xtr,ytr); ptr=prof.transform(Xtr); pte=prof.transform(Xte); ok('ZoneProfileEncoder',{'train_shape':list(ptr.shape),'test_shape':list(pte.shape)})
 r=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=10))]).fit(ptr,ytr); ok('ZoneProfileRidge',reg('Ridge zone profiles',yte,r.predict(pte)))
except Exception as e: fail('ZoneProfileEncoder',e)

try:
 dt=DepthTransformer(columns=num,group_name='weather'); dtr=dt.fit_transform(Xtr[num]); dte=dt.transform(Xte[num]); ok('DepthTransformer',{'shape':list(dtr.shape)})
 cd=CategoricalDepthTransformer(columns=['season','mnth','hr','weekday','workingday'],group_name='calendar'); ctr=cd.fit_transform(Xtr[cat]); cte=cd.transform(Xte[cat]); ok('CategoricalDepthTransformer',{'shape':list(ctr.shape)})
 etr=pd.concat([dtr.filter(like='__coreness'),ctr.filter(like='__coreness')],axis=1); ete=pd.concat([dte.filter(like='__coreness'),cte.filter(like='__coreness')],axis=1); dc=DepthCrowd(columns=list(etr.columns),rank_normalize=True,vote_threshold=.05,group_name='crowd'); crtr=dc.fit_transform(etr); crte=dc.transform(ete); ok('DepthCrowd',{'shape':list(crtr.shape),'numeric_columns':list(crtr.select_dtypes(include=np.number).columns)})
except Exception as e: fail('DepthPipeline',e)

try:
 cg=ConditionalZoneGrid(columns=['temp','hum'],segment_columns=['workingday']).fit(Xtr[['temp','hum','workingday']],ytr); cgt=cg.transform(Xte[['temp','hum','workingday']]); ok('ConditionalZoneGrid',{'shape':list(cgt.shape)})
except Exception as e: fail('ConditionalZoneGrid',e)

try:
 zfs=ZoneFeatureSpace(zone_profiles=['temp','hum','hr'],depth_scores=[('weather',num)],categorical_depth_scores=True,conditional_grids=[('temp','hum',['workingday'])],random_state=42).fit(Xtr,ytr); ftr=zfs.transform(Xtr); fte=zfs.transform(Xte); ok('ZoneFeatureSpace',{'train_shape':list(ftr.shape),'test_shape':list(fte.shape)})
except Exception as e: fail('ZoneFeatureSpace',e)

try:
 threshold=float(ytr.quantile(.75)); cytr=(ytr>=threshold).astype(int); cyte=(yte>=threshold).astype(int); clf=ZoneBoostClassifier(n_rounds=80,learning_rate=.05,categorical_features=cat,max_pair_interactions=10,validation_fraction=.2,n_iter_no_change=12,calibrate=True,calibration_fraction=.1,refit_on_full_data=True,random_state=42).fit(Xtr,cytr); prob=clf.predict_proba(Xte)[:,1]; pred=clf.predict(Xte); cm=confusion_matrix(cyte,pred); pd.DataFrame(cm).to_csv(OUT/'classification_confusion_matrix.csv',index=False); ok('ZoneBoostClassifier',{'threshold':threshold,'accuracy':float(accuracy_score(cyte,pred)),'balanced_accuracy':float(balanced_accuracy_score(cyte,pred)),'macro_F1':float(f1_score(cyte,pred,average='macro')),'ROC_AUC':float(roc_auc_score(cyte,prob)),'log_loss':float(log_loss(cyte,np.c_[1-prob,prob]))})
except Exception as e: fail('ZoneBoostClassifier',e)

try:
 m11=df.yr.astype(int)==0; m12=df.yr.astype(int)==1; old=ZoneBoostRegressor(**base).fit(df.loc[m11,features],y[m11]); new=ZoneBoostRegressor(**base).fit(df.loc[m12,features],y[m12]); drift=compare_models(old,new,df.loc[m12,features],y[m12]); json.dump(drift,open(OUT/'drift.json','w'),default=str,indent=2); ok('compare_models',{'type':type(drift).__name__}); ok('flag_drift',flag_drift(old,new,df.loc[m12,features],y[m12],alpha=.1))
except Exception as e: fail('Drift',e)

try:
 ts=ZoneBoostTimeSeries(base_estimator=ZoneBoostRegressor(n_rounds=30,learning_rate=.05,categorical_features=cat,max_pair_interactions=5,random_state=42),time_col='dteday',freq='Q',window='expanding',min_periods=2,random_state=42).fit(df[['dteday']+features],y); sr=ts.stability_report(); pd.DataFrame(sr).to_csv(OUT/'time_series_stability.csv',index=False); ok('ZoneBoostTimeSeries',{'shape':list(pd.DataFrame(sr).shape)})
except Exception as e: fail('ZoneBoostTimeSeries',e)

try:
 bs=BootstrapStability(ZoneBoostRegressor(n_rounds=25,learning_rate=.05,categorical_features=cat,max_pair_interactions=5,random_state=42),n_bootstrap=4,random_state=42).fit(Xtr,ytr); inc=bs.inclusion_frequency(); inc.to_csv(OUT/'bootstrap_inclusion.csv',header=['frequency']); ok('BootstrapStability',{'terms':len(inc)})
except Exception as e: fail('BootstrapStability',e)

try:
 sm=ZoneBoostRegressor(n_rounds=15,learning_rate=.05,categorical_features=cat,max_pair_interactions=5,random_state=42).fit(Xtr,ytr); sql=compile_to_sql(sm,table_name='bike_scoring'); open(OUT/'model.sql','w').write(sql); ok('compile_to_sql',{'characters':len(sql)})
except Exception as e: fail('compile_to_sql',e)

pd.DataFrame(metrics).sort_values('RMSE').to_csv(OUT/'model_metrics.csv',index=False); summary['metrics']=metrics; summary['functionality']=status; json.dump(summary,open(OUT/'summary.json','w'),indent=2,default=str); print(json.dumps(summary,indent=2,default=str))
