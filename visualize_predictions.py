import torch 
import numpy as np 
import matplotlib .pyplot as plt 
from pina import LabelTensor 
from run_pina_model import SignalingModel 
from data_utils import prepare_training_tensors ,TRAINING_DATA_LIST ,SPECIES_ORDER 

def visualize ():
    print ("Loading data and model...")
    train_data ,test_data ,scalers =prepare_training_tensors (
    split_mode ="cutoff",
    train_until_hour =48.0,
    normalization_mode ="train_only",
    )

    model =SignalingModel ()
    model .load_state_dict (torch .load ("pina_signaling_model_Vemurafenib_Only.pth",map_location ='cpu',weights_only =True ))
    model .eval ()

    y_min =scalers ['y_min']
    y_range =scalers ['y_range']
    t_max =48.0 

    plt .style .use ('ggplot')
    plt .rcParams ['font.family']='sans-serif'

    selected_conditions =[
    'Vemurafenib Only (0.5)'
    ]

    plot_species = SPECIES_ORDER
    species_indices =[SPECIES_ORDER .index (s )for s in plot_species ]

    for cond_name in selected_conditions :

        exp_data =next ((e for e in TRAINING_DATA_LIST if e ['name']==cond_name ),None )
        if not exp_data :
            continue 

        print (f"Visualizing condition: {cond_name }")

        drugs_val =[
        exp_data ['drugs']['vemurafenib'],
        exp_data ['drugs']['trametinib'],
        exp_data ['drugs']['pi3k_inhibitor'],
        exp_data ['drugs']['ras_inhibitor']
        ]

        t_dense =np .linspace (0 ,t_max ,200 ).astype (np .float32 )
        t_norm =(t_dense /t_max ).reshape (-1 ,1 )

        drugs_mat =np .tile (drugs_val ,(len (t_dense ),1 )).astype (np .float32 )
        X_input =LabelTensor (
        torch .cat ([torch .tensor (t_norm ),torch .tensor (drugs_mat )],dim =1 ),
        ['t','vem','tram','pi3k','ras']
        )

        with torch .no_grad ():
            preds_norm =model (X_input ).as_subclass (torch .Tensor )
            preds =preds_norm *y_range +y_min 

        fig ,axes =plt .subplots (2 ,5 ,figsize =(20 ,8 ),sharex =True )
        axes_flat = axes.flatten()
        fig .suptitle (f"Pathway Dynamics: {cond_name }",fontsize =16 ,fontweight ='bold',y =1.05 )

        for i ,(sp_name ,sp_idx )in enumerate (zip (plot_species ,species_indices )):
            ax =axes_flat [i ]

            ax .plot (t_dense ,preds [:,sp_idx ],label ='PINN Prediction',color ='#1f77b4',linewidth =3 ,alpha =0.8 )

            t_true =exp_data ['time_points']
            y_true =exp_data ['species'][sp_name ]

            if cond_name =="Vem + PI3Ki Combo":
                is_test =np .array ([abs (t -24.0 )<1e-4 for t in t_true ])
            else :
                is_test =np .zeros_like (t_true ,dtype =bool )

            ax .scatter (t_true [~is_test ],y_true [~is_test ],color ='#2ca02c',s =80 ,label ='Train Data',edgecolors ='white',zorder =5 )
            if is_test .any ():
                ax .scatter (t_true [is_test ],y_true [is_test ],color ='#d62728',s =100 ,marker ='*',label ='Test Data',zorder =6 )

            ax .set_title (sp_name ,fontsize =14 ,fontweight ='semibold')
            ax .set_xlabel ("Time (hours)")
            if i ==0 :
                ax .set_ylabel ("Expression Level")

            ax .set_ylim (bottom =0 )
            ax .set_ylim (top =max (y_true .max ()*1.5 ,preds [:,sp_idx ].max ()*1.5 ,0.5 ))

            if i ==len (plot_species )-1 :
                ax .legend (loc ='upper right',frameon =True ,fontsize =10 )

        plt .tight_layout ()
        filename =f"plot_{cond_name .replace (' ','_').replace ('(','').replace (')','')}.png"
        plt .savefig (filename ,bbox_inches ='tight',dpi =150 )
        print (f"  Saved plot to {filename }")
        plt .close ()

if __name__ =="__main__":
    visualize ()

