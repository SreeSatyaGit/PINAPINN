from typing import Dict ,List ,Tuple ,Optional ,Any 
import logging 
import torch 
import numpy as np 
from torch .utils .data import Dataset 

ExperimentalCondition =Dict [str ,Any ]
SpeciesData =Dict [str ,np .ndarray ]
LOGGER =logging .getLogger (__name__ )

SPECIES_ORDER :List [str ]=[
'pEGFR','HER2','HER3','IGF1R','pCRAF','pMEK',
'pERK','DUSP6','pAKT','p4EBP1'
]

def _compute_basal_steady_state ()->Dict [str ,np .ndarray ]:
    """
    Compute the average t=0 (pre-treatment) protein levels across all
    experimental conditions. These represent the constitutive steady state.

    Returns a dict of {species_name: scalar_value} for the basal level.
    """
    _t0_values ={
    'pEGFR':[0.291928893 ,0.35120201 ,0.222379739 ,0.333114406 ,0.641846 ],
    'HER2':[0.245236744 ,0.328412641 ,0.306924546 ,0.411424 ,0.745302 ],
    'HER3':[0.203233765 ,0.297534144 ,0.295284147 ,0.411424 ,0.745302 ],
    'IGF1R':[0.96024414 ,1.193833833 ,1.180034579 ,1.365603 ,1.179241 ],
    'pCRAF':[0.366397596 ,0.233819043 ,0.234376572 ,0.989486 ,2.764099 ],
    'pMEK':[1.75938884 ,2.024063616 ,1.936660577 ,2.018304 ,2.462343 ],
    'pERK':[2.903209735 ,3.487158237 ,3.273353557 ,3.47456 ,4.450806 ],
    'DUSP6':[2.677161325 ,3.054033606 ,2.854207662 ,1.47878 ,1.519079 ],
    'pAKT':[1.080071 ,0.549427631 ,0.527301325 ,0.320037 ,0.536743 ],
    'p4EBP1':[1.002468056 ,0.984913436 ,0.793559668 ,0.698339 ,0.74627 ],
    }
    return {sp :np .mean (vals )for sp ,vals in _t0_values .items ()}

_BASAL_SS =_compute_basal_steady_state ()

_NO_DRUG_TIME_POINTS =np .array ([0.0 ],dtype =np .float32 )
_NO_DRUG_SPECIES ={
sp :np .array ([_BASAL_SS [sp ]],dtype =np .float32 )
for sp in SPECIES_ORDER 
}

TRAINING_DATA_LIST :List [Dict [str ,Any ]]=[
{
'name':'No Drug (Basal Steady State)',
'time_points':_NO_DRUG_TIME_POINTS ,
'species':_NO_DRUG_SPECIES ,
'drugs':{'vemurafenib':0.0 ,'trametinib':0.0 ,'pi3k_inhibitor':0.0 ,'ras_inhibitor':0.0 }
},
{
'name':'Vemurafenib Only (0.5)',
'time_points':np .array ([0 ,1 ,4 ,8 ,24 ,48 ]),
'species':{
'pEGFR':np .array ([0.291928893 ,0.392400458 ,0.265016688 ,0.394238749 ,0.006158316 ,0.008115099 ]),
'HER2':np .array ([0.245236744 ,0.177917339 ,0.239075259 ,0.306884773 ,1.066654783 ,1.005085151 ]),
'HER3':np .array ([0.203233765 ,0.194358998 ,0.303475212 ,0.674083831 ,0.89702403 ,0.459831389 ]),
'IGF1R':np .array ([0.96024414 ,1.070624757 ,1.126551098 ,1.157072001 ,0.734617533 ,0.4579226222 ]),
'pCRAF':np .array ([0.366397596 ,0.537106733 ,0.465541704 ,0.586732657 ,1.102322681 ,0.269181259 ]),
'pMEK':np .array ([1.75938884 ,0.170160085 ,0.095112609 ,0.201000276 ,0.219207054 ,0.502831668 ]),
'pERK':np .array ([2.903209735 ,0.207867788 ,0.303586121 ,0.805254439 ,1.408362153 ,1.847606441 ]),
'DUSP6':np .array ([2.677161325 ,2.782754577 ,1.130758062 ,0.395642757 ,0.828575853 ,0.916618219 ]),
'pAKT':np .array ([1.080071 ,0.900024 ,1.066681 ,0.899337 ,1.21576 ,1.333254 ]),
'p4EBP1':np .array ([1.002468056 ,1.276793699 ,1.252681407 ,1.707504483 ,1.271216967 ,0.61389625 ])
},
'drugs':{'vemurafenib':0.5 ,'trametinib':0.0 ,'pi3k_inhibitor':0.0 ,'ras_inhibitor':0.0 }
},
{
'name':'Trametinib Only (0.3)',
'time_points':np .array ([0 ,1 ,4 ,8 ,24 ,48 ]),
'species':{
'pEGFR':np .array ([0.35120201 ,0.545885712 ,0.642780201 ,0.359436744 ,0.009710876 ,0.01311083 ]),
'HER2':np .array ([0.328412641 ,0.302493913 ,0.32569075 ,0.254737487 ,1.050977922 ,1.026829681 ]),
'HER3':np .array ([0.297534144 ,0.20792856 ,0.227078284 ,0.297261013 ,0.966331288 ,0.97673174 ]),
'IGF1R':np .array ([1.193833833 ,1.170396382 ,1.060073053 ,0.983950361 ,0.525472084 ,0.298723901 ]),
'pCRAF':np .array ([0.233819043 ,0.451384259 ,0.355935204 ,0.786954922 ,0.946694075 ,0.26220219 ]),
'pMEK':np .array ([2.024063616 ,0.501888368 ,0.442535778 ,0.508820348 ,0.671938823 ,0.983710872 ]),
'pERK':np .array ([3.487158237 ,0.114217722 ,0.022162422 ,0.03726381 ,0.40696761 ,1.204092044 ]),
'DUSP6':np .array ([3.054033606 ,3.118384551 ,1.178142126 ,0.367382855 ,0.031574513 ,0.092390719 ]),
'pAKT':np .array ([0.549427631 ,0.642783939 ,1.046735362 ,0.944355203 ,0.479593107 ,0.310063914 ]),
'p4EBP1':np .array ([0.984913436 ,1.263725259 ,1.355812598 ,1.380725265 ,0.941991484 ,0.281153536 ])
},
'drugs':{'vemurafenib':0.0 ,'trametinib':0.3 ,'pi3k_inhibitor':0.0 ,'ras_inhibitor':0.0 }
},

{
'name':'Vem + Tram Combo',
'time_points':np .array ([0 ,1 ,4 ,8 ,24 ,48 ]),
'species':{
'pEGFR':np .array ([0.222379739 ,0.622877159 ,0.629217784 ,0.533530834 ,0.022513609 ,0.010036399 ]),
'HER2':np .array ([0.306924546 ,0.275751955 ,0.32171108 ,0.23070312 ,1.013023288 ,1.045536401 ]),
'HER3':np .array ([0.295284147 ,0.285719072 ,0.385045943 ,0.582261781 ,0.751301308 ,0.264889608 ]),
'IGF1R':np .array ([1.180034579 ,0.967927178 ,0.808905442 ,0.781013289 ,0.41928501 ,0.870763253 ]),
'pCRAF':np .array ([0.234376572 ,0.641878896 ,0.567434544 ,0.406320223 ,0.582899195 ,0.25113447 ]),
'pMEK':np .array ([1.936660577 ,0.029380652 ,0.012873835 ,0.03390921 ,0.095155796 ,0.944936578 ]),
'pERK':np .array ([3.273353557 ,0.075717978 ,0.011570416 ,0.00642985 ,0.041863585 ,0.91621491 ]),
'DUSP6':np .array ([2.854207662 ,2.842703936 ,1.163746208 ,0.332720449 ,0.030434242 ,0.094073888 ]),
'pAKT':np .array ([0.527301325 ,0.614645732 ,0.95895017 ,0.895019432 ,0.412820453 ,0.269891704 ]),
'p4EBP1':np .array ([0.793559668 ,1.176099875 ,1.210864904 ,1.415698564 ,0.858543042 ,0.167293554 ])
},
'drugs':{'vemurafenib':0.5 ,'trametinib':0.3 ,'pi3k_inhibitor':0.0 ,'ras_inhibitor':0.0 }
},
{
'name':'Vem + PI3Ki Combo',
'time_points':np .array ([0 ,1 ,4 ,8 ,24 ,48 ]),
'species':{
'pEGFR':np .array ([0.333114406 ,0.304726551 ,0.329135261 ,0.368629137 ,0.15272344 ,0.056896369 ]),
'HER2':np .array ([0.411424 ,0.55907 ,0.526718 ,0.938131 ,1.064491 ,2.521235 ]),
'HER3':np .array ([0.411424 ,0.55907 ,0.526718 ,0.938131 ,1.064491 ,2.521235 ]),
'IGF1R':np .array ([1.365603 ,1.141697 ,1.536194 ,1.383503 ,1.438155 ,1.746471 ]),
'pCRAF':np .array ([0.989486 ,0.600938 ,0.567638 ,0.406382 ,0.228887 ,0.49824 ]),
'pMEK':np .array ([2.018304 ,0.340262 ,0.812539 ,0.878866 ,0.478651 ,0.899268 ]),
'pERK':np .array ([3.47456 ,0 ,2.286908 ,2.961433 ,2.12377 ,3.14885 ]),
'DUSP6':np .array ([1.47878 ,1.186021 ,0.677175 ,0.826286 ,0.453723 ,1.852728 ]),
'pAKT':np .array ([0.320037 ,0.026226 ,0.063851 ,0.05965 ,0.015918 ,0.09892 ]),
'p4EBP1':np .array ([0.698339 ,0.175174 ,0.146055 ,0.10574 ,0.198613 ,0.131825 ])
},
'drugs':{'vemurafenib':0.5 ,'trametinib':0.0 ,'pi3k_inhibitor':0.5 ,'ras_inhibitor':0.0 }
},
{
'name':'Vem + panRAS Combo',
'time_points':np .array ([0 ,1 ,4 ,8 ,24 ,48 ]),
'species':{
'pEGFR':np .array ([0.641846 ,0.40072 ,0.16665 ,0.265356 ,0.338174 ,0.072136 ]),
'HER2':np .array ([0.745302 ,0.421645 ,0.527132 ,0.349948 ,0.436277 ,0.299083 ]),
'HER3':np .array ([0.745302 ,0.421645 ,0.527132 ,0.349948 ,0.436277 ,0.299083 ]),
'IGF1R':np .array ([1.179241 ,0.97553 ,0.59481 ,0.710898 ,0.607603 ,0.62405 ]),
'pCRAF':np .array ([2.764099 ,2.230603 ,3.58039 ,2.311266 ,2.405439 ,1.47887 ]),
'pMEK':np .array ([2.462343 ,0.156141 ,0.599231 ,1.118925 ,0.964204 ,0.73809 ]),
'pERK':np .array ([4.450806 ,0 ,0.192028 ,0.213346 ,0.534783 ,3.47456 ]),
'DUSP6':np .array ([1.519079 ,0.420432 ,0 ,0.173392 ,0.068671 ,0.181719 ]),
'pAKT':np .array ([0.536743 ,1.171471 ,1.100104 ,0.904367 ,0.415397 ,1.076115 ]),
'p4EBP1':np .array ([0.74627 ,1.027582 ,0.89376 ,0.876078 ,0.41763 ,0.215742 ])
},
'drugs':{'vemurafenib':0.5 ,'trametinib':0.0 ,'pi3k_inhibitor':0.0 ,'ras_inhibitor':0.5 }
}
]

class SignalingDataset (Dataset ):
    """
    Standard PyTorch dataset for experimental signaling data.
    Maps (time, drugs) -> protein_expression.
    """
    def __init__ (self ,t :torch .Tensor ,drugs :torch .Tensor ,y :torch .Tensor ):
        self .t =t 
        self .drugs =drugs 
        self .y =y 

    def __len__ (self )->int :
        return len (self .t )

    def __getitem__ (self ,idx :int )->Tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ]:
        return self .t [idx ],self .drugs [idx ],self .y [idx ]

def prepare_training_tensors (
train_until_hour :float =48.0 ,
condition_name :Optional [str ]=None ,
split_mode :str ="holdout",
holdout_timepoints :Optional [List [float ]]=None ,
holdout_condition :Optional [str ]=None ,
partial_condition_train_timepoints :Optional [List [float ]]=None ,
normalization_mode :str ="train_only",
)->Tuple [Dict [str ,Any ],Dict [str ,Any ],Dict [str ,torch .Tensor ]]:
    """
    Prepares training and testing tensors from raw experimental data.

    Args:
        train_until_hour: Cutoff time for training (used in 'cutoff' mode).
        condition_name: If provided, filters to only this condition.
        split_mode: 'cutoff'                    = train on t <= train_until_hour,
                    'holdout'                   = leave specific timepoints out for testing,
                    'condition_holdout'         = leave entire named condition out for testing,
                    'partial_condition_holdout' = train on early timepoints of holdout_condition
                                                  only, predict remaining timepoints of that
                                                  condition. All other conditions train in full.
        holdout_timepoints: List of timepoints (in hours) to hold out for testing.
                           Only used when split_mode='holdout'. Default: [8.0, 24.0].
        holdout_condition: Name of the condition to hold out entirely.
                           Only used when split_mode='condition_holdout'.
                           Must exactly match a 'name' field in TRAINING_DATA_LIST.
        partial_condition_train_timepoints: Timepoints (hours) to INCLUDE in training for
            holdout_condition. All other timepoints of that condition become the test set.
            Only used when split_mode='partial_condition_holdout'.
            Example: [0.0, 4.0] trains on t=0 and t=4 of holdout_condition,
            holds out t=1, 8, 24, 48.

    Returns:
        train_data: Dict with keys 't', 'drugs', 'y_norm', 'y_raw'
        test_data: Dict with keys 't', 'drugs', 'y_norm', 'y_raw'
        scalers: Dict with normalization factors
    """
    LOGGER .info (
    "Preparing tensors | split_mode=%s | holdout_condition=%s | normalization_mode=%s",
    split_mode ,
    holdout_condition ,
    normalization_mode ,
    )

    if holdout_timepoints is None :
        holdout_timepoints =[24.0 ]

    all_t ,all_y ,all_drugs ,all_condition_labels =[],[],[],[]

    experiments =TRAINING_DATA_LIST 
    if condition_name :
        experiments =[e for e in TRAINING_DATA_LIST if e ['name']==condition_name ]
        if not experiments :
            raise ValueError (f"Condition '{condition_name }' not found.")

    for exp in experiments :
        t_points =exp ['time_points'].astype (np .float32 )
        num_pts =len (t_points )

        y_exp =np .zeros ((num_pts ,10 ),dtype =np .float32 )
        for i ,species in enumerate (SPECIES_ORDER ):
            y_exp [:,i ]=exp ['species'][species ]

        drug_vec =np .array ([
        exp ['drugs']['vemurafenib'],
        exp ['drugs']['trametinib'],
        exp ['drugs']['pi3k_inhibitor'],
        exp ['drugs']['ras_inhibitor']
        ],dtype =np .float32 )
        drug_mat =np .tile (drug_vec ,(num_pts ,1 ))

        all_t .append (t_points )
        all_y .append (y_exp )
        all_drugs .append (drug_mat )
        all_condition_labels .append (np .array ([exp ["name"]]*num_pts ))

    t_data =np .concatenate (all_t )
    y_data =np .concatenate (all_y )
    drugs_data =np .concatenate (all_drugs )
    condition_data =np .concatenate (all_condition_labels )

    t_max =48.0 

    if split_mode =="holdout":
        test_mask =np .array ([
        any (abs (float (t )-h )<1e-4 for h in holdout_timepoints )
        for t in t_data 
        ])
        train_mask =~test_mask 
    elif split_mode =="condition_holdout":
        if holdout_condition is None :
            raise ValueError (
            "split_mode='condition_holdout' requires holdout_condition to be specified. "
            f"Available conditions: {[e ['name']for e in TRAINING_DATA_LIST ]}"
            )
        available =[e ['name']for e in TRAINING_DATA_LIST ]
        if holdout_condition not in available :
            raise ValueError (
            f"holdout_condition='{holdout_condition }' not found. "
            f"Available conditions: {available }"
            )

        condition_labels =[]
        for exp in experiments :
            condition_labels .extend ([exp ['name']]*len (exp ['time_points']))
        condition_labels =np .array (condition_labels )
        test_mask =(condition_labels ==holdout_condition )
        train_mask =~test_mask 
    elif split_mode =="partial_condition_holdout":
        if holdout_condition is None :
            raise ValueError (
            "split_mode='partial_condition_holdout' requires holdout_condition "
            "to be specified. "
            f"Available conditions: {[e ['name']for e in TRAINING_DATA_LIST ]}"
            )
        if partial_condition_train_timepoints is None :
            raise ValueError (
            "split_mode='partial_condition_holdout' requires "
            "partial_condition_train_timepoints to be specified. "
            "Example: [0.0, 4.0]"
            )
        available =[e ['name']for e in TRAINING_DATA_LIST ]
        if holdout_condition not in available :
            raise ValueError (
            f"holdout_condition='{holdout_condition }' not found. "
            f"Available conditions: {available }"
            )

        condition_labels =[]
        for exp in experiments :
            condition_labels .extend ([exp ['name']]*len (exp ['time_points']))
        condition_labels =np .array (condition_labels )

        train_t_set =set (partial_condition_train_timepoints )

        train_mask =np .array ([
        True if label !=holdout_condition 
        else any (abs (float (t )-anchor )<1e-4 for anchor in train_t_set )
        for label ,t in zip (condition_labels ,t_data )
        ])
        test_mask =~train_mask 
    elif split_mode =="cutoff":
        train_mask =t_data <=train_until_hour 
        test_mask =~train_mask 
    else :
        raise ValueError (
        f"Invalid split_mode: {split_mode }. "
        "Valid modes: ['holdout', 'condition_holdout', 'partial_condition_holdout', 'cutoff']"
        )

    def compute_scaler (y_train :np .ndarray ,y_full :np .ndarray ,mode :str ="train_only")->Tuple [np .ndarray ,np .ndarray ]:
        if mode =="train_only":
            y_min_local =np .min (y_train ,axis =0 )
            y_max_local =np .max (y_train ,axis =0 )
        elif mode =="global":
            y_min_local =np .min (y_full ,axis =0 )
            y_max_local =np .max (y_full ,axis =0 )
        else :
            raise ValueError (f"Invalid normalization mode: {mode }. Valid modes: ['train_only', 'global']")
        return y_min_local ,y_max_local 

    if not train_mask .any ():
        raise ValueError ("Training split is empty. Check split_mode and holdout configuration.")

    y_min ,y_max =compute_scaler (y_data [train_mask ],y_data ,mode =normalization_mode )
    y_range =y_max -y_min 
    y_range [y_range ==0 ]=1.0 

    y_norm_all =(y_data -y_min )/y_range 
    y_norm_all =np .nan_to_num (y_norm_all ,nan =0.0 ,posinf =1.0 ,neginf =0.0 )

    def package_data (mask ):
        return {
        't':t_data [mask ],
        't_norm':(t_data [mask ]/t_max ).reshape (-1 ,1 ),
        'drugs':drugs_data [mask ],
        'y_norm':y_norm_all [mask ],
        'y_raw':y_data [mask ],
        'condition':condition_data [mask ],
        }

    train_data =package_data (train_mask )
    test_data =package_data (test_mask )
    LOGGER .info (
    "Prepared dataset sizes | train=%d | test=%d",
    len (train_data ['t']),
    len (test_data ['t']),
    )

    scalers ={
    'y_min':torch .tensor (y_min ,dtype =torch .float32 ),
    'y_range':torch .tensor (y_range ,dtype =torch .float32 ),
    't_range':torch .tensor (t_max ,dtype =torch .float32 ),
    'normalization_mode':normalization_mode ,
    }

    return train_data ,test_data ,scalers 

def get_collocation_points (n_points :int =2000 ,no_drug_fraction :float =0.2 )->Tuple [torch .Tensor ,torch .Tensor ]:
    """
    Generates physics collocation points covering the domain [0, 48].
    Drug concentrations are sampled from actual training conditions with
    small perturbations, ensuring physics is enforced near real data.

    A guaranteed fraction of points are pure no-drug (all zeros) to enforce
    the steady-state constraint: dX/dt ≈ 0 when no drug is applied.

    Args:
        n_points: Total points to generate.
        no_drug_fraction: Fraction of points guaranteed to be no-drug (default 0.2).

    Returns:
        t_physics: Temporal points (normalized).
        drugs_physics: Drug concentration points.
    """

    t_early_fraction =0.60 
    n_early =int (n_points *t_early_fraction )
    n_late =n_points -n_early 
    t_early =np .random .uniform (0.0 ,0.167 ,size =(n_early ,1 )).astype (np .float32 )
    t_late =np .random .uniform (0.167 ,1.0 ,size =(n_late ,1 )).astype (np .float32 )
    t_physics =np .concatenate ([t_early ,t_late ],axis =0 )

    actual_conditions =np .array ([
    [exp ['drugs']['vemurafenib'],exp ['drugs']['trametinib'],
    exp ['drugs']['pi3k_inhibitor'],exp ['drugs']['ras_inhibitor']]
    for exp in TRAINING_DATA_LIST 
    ],dtype =np .float32 )

    n_no_drug =int (n_points *no_drug_fraction )
    n_other =n_points -n_no_drug 

    idx =np .random .choice (len (actual_conditions ),size =n_other )
    drugs_other =actual_conditions [idx ].copy ()

    perturbation =np .random .normal (0 ,0.05 ,size =(n_other ,4 )).astype (np .float32 )

    no_drug_rows =(drugs_other .sum (axis =1 )<1e-6 )
    perturbation [no_drug_rows ]=0.0 

    drugs_other +=perturbation 
    drugs_other =np .clip (drugs_other ,0.0 ,None ).astype (np .float32 )

    drugs_no_drug =np .zeros ((n_no_drug ,4 ),dtype =np .float32 )

    drugs_phys =np .concatenate ([drugs_no_drug ,drugs_other ],axis =0 )

    shuffle_idx =np .random .permutation (n_points )
    t_physics =t_physics [shuffle_idx ]
    drugs_phys =drugs_phys [shuffle_idx ]

    return torch .tensor (t_physics ),torch .tensor (drugs_phys )

