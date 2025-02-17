import os
import pandas as pd
import matplotlib.pyplot as plt

base_directory = "./predictions"
# Using os.walk
for root, dirs, files in os.walk(base_directory):
    for file in files:
        print(os.path.join(root, file))
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path) 
     
        sample_title = file_path.split('/')[-1].split('_predictions')[0]
        sample_title = 'Sample ' + sample_title
    
        types = {10,20,30,40,50,60}
        # 10 = CD8
        # 20 = CD4
        # 30 = mhcII
        # 40 = pd1
        # 50 = pd1tcf
        # 60 = tcf

        data_plot1 = data[~(data['Predictions'] == 20)] #drop CD4
        data_plot1 = data_plot1[~(data['Predictions'] == 30)] #drop MHC II
        data_plot1['Predictions'] = data_plot1['Predictions'].replace(60, 10) #replace tcf with cd8

        data_plot2 = data[data['Predictions'] == 30] #MHC II+ only

        data_plot3 = data[data['Predictions'] == 30] #MHC II+ only  testing

        # Create a dictionary to map types to colors
        color_map_plot1 = {10: 'lightgray', 40: 'red', 50: '#39FF14'}
        color_map_plot2 = {30: 'gray'}
        color_map_plot3 = {30: 'blue'} #testing

        colors_plot1 = data_plot1['Predictions'].map(color_map_plot1)
        colors_plot2 = data_plot2['Predictions'].map(color_map_plot2)
        colors_plot3 = data_plot3['Predictions'].map(color_map_plot3) #testing

        # create grid
        fig = plt.figure(layout=None, figsize=(30,11)) 
        # fig = plt.rcParams.update({'font.size': 20})
        gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0.2, wspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])

        ############### plot 1 ####################################################################################################################
        ax0.scatter(data_plot1['Centroid.X.µm'], data_plot1['Centroid.Y.µm']*(-1), s=4/100, c=colors_plot1, marker='.')
        # ax0.set_title(sample_title, fontsize=30,color='red', fontweight='bold',loc='left')  # Add title to the first subplot
        ax0.set_xlabel("Centroid X µm", fontsize=10)
        ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=4, va='center')
        ax0.set_ylabel("Centroid Y µm", fontsize=10)
        ax0.set_yticklabels(ax0.get_yticklabels(), rotation=90, fontsize=4, va='center')
        ax0.legend(handles=[plt.Line2D([], [], color='gray', marker='o', label='CD8+ Cells'),
                            plt.Line2D([], [], color='red', marker='o', label='CD8+PD1+TCF- Cells'),
                            plt.Line2D([], [], color='#39FF14', marker='o', label='CD8+PD1+TCF+ Cells')],
                            fontsize=8)
        ############### plot 2 ####################################################################################################################
        ax1.scatter(data_plot2['Centroid.X.µm'], data_plot2['Centroid.Y.µm']*(-1), s=4/100,c=colors_plot2, marker='.')
        ax1.set_xlabel("Centroid X µm", fontsize=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=4, va='center')
        ax1.set_ylabel("Centroid Y µm", fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=90, fontsize=4, va='center')
        ax1.legend(handles=[plt.Line2D([], [], color='gray', marker='o', label='MHC II+')], fontsize=8)
        ############### plot 3 ####################################################################################################################
        
        ax2.scatter(data_plot2['Centroid.X.µm'], data_plot2['Centroid.Y.µm']*(-1), s=4/100,c=colors_plot2, marker='.')
        ax2.set_xlabel("Centroid X µm", fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=4, va='center')
        ax2.set_ylabel("Centroid Y µm", fontsize=10)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=90, fontsize=4, va='center')
        ax2.legend(handles=[plt.Line2D([], [], color='gray', marker='o', label='MHC II+')], fontsize=8)
        ############### plot 4 ####################################################################################################################
        #  cd8 = 10
        data_cd8 = pd.concat([data[data['Predictions'] == 10],data[data['Predictions'] == 40],data[data['Predictions'] == 50],data[data['Predictions'] == 60]],ignore_index=True)
        cd8_max_x = data['Centroid.X.µm'].max() 
        cd8_min_x = data['Centroid.X.µm'].min()
        cd8_max_y = data['Centroid.Y.µm'].max() 
        cd8_min_y = data['Centroid.Y.µm'].min() 
        cd8_area = ((cd8_max_x - cd8_min_x) * (cd8_max_y - cd8_min_y))  // 10000  # convert to mm^2
        cd8_count = data_cd8.shape[0]
        cd8_mm2 = cd8_count / cd8_area

         #  mhcII = 30
        data_mhcII = data[data['Predictions'] == 30]
        mhcII_max_x = data['Centroid.X.µm'].max() 
        mhcII_min_x = data['Centroid.X.µm'].min()
        mhcII_max_y = data['Centroid.Y.µm'].max() 
        mhcII_min_y = data['Centroid.Y.µm'].min() 
        mhcII_area = ((mhcII_max_x - mhcII_min_x) * (mhcII_max_y - mhcII_min_y))  // 10000  # convert to mm^2
        mhcII_count = data_mhcII.shape[0]
        mhcII_mm2 = mhcII_count / mhcII_area

         #  cd4 = 20
        data_cd4 = data[data['Predictions'] == 30]
        cd4_max_x = data['Centroid.X.µm'].max() 
        cd4_min_x = data['Centroid.X.µm'].min()
        cd4_max_y = data['Centroid.Y.µm'].max() 
        cd4_min_y = data['Centroid.Y.µm'].min() 
        cd4_area = ((cd4_max_x - cd4_min_x) * (cd4_max_y - cd4_min_y))  // 10000  # convert to mm^2
        cd4_count = data_cd4.shape[0]
        cd4_mm2 = cd4_count / cd4_area

         #  tcf = 60,50
        data_tcf = pd.concat([data[data['Predictions'] == 60],data[data['Predictions'] == 50]],ignore_index=True)
        tcf_max_x = data['Centroid.X.µm'].max() 
        tcf_min_x = data['Centroid.X.µm'].min()
        tcf_max_y = data['Centroid.Y.µm'].max() 
        tcf_min_y = data['Centroid.Y.µm'].min() 
        tcf_area = ((tcf_max_x - tcf_min_x) * (tcf_max_y - tcf_min_y))  // 10000  # convert to mm^2
        tcf_count = data_tcf.shape[0]
        tcf_mm2 = tcf_count / tcf_area

         #  pd1 = 40,50
        data_pd1 = pd.concat([data[data['Predictions'] == 40],data[data['Predictions'] == 50]],ignore_index=True)
        pd1_max_x = data['Centroid.X.µm'].max() 
        pd1_min_x = data['Centroid.X.µm'].min()
        pd1_max_y = data['Centroid.Y.µm'].max() 
        pd1_min_y = data['Centroid.Y.µm'].min() 
        pd1_area = ((pd1_max_x - pd1_min_x) * (pd1_max_y - pd1_min_y))  // 10000  # convert to mm^2
        pd1_count = data_pd1.shape[0]
        pd1_mm2 = pd1_count / pd1_area

         #  pd1tcf = 50
        data_pd1tcf = data[data['Predictions'] == 30]
        pd1tcf_max_x = data['Centroid.X.µm'].max() 
        pd1tcf_min_x = data['Centroid.X.µm'].min()
        pd1tcf_max_y = data['Centroid.Y.µm'].max() 
        pd1tcf_min_y = data['Centroid.Y.µm'].min() 
        pd1tcf_area = ((pd1tcf_max_x - pd1tcf_min_x) * (pd1tcf_max_y - pd1tcf_min_y))  // 10000  # convert to mm^2
        pd1tcf_count = data_pd1tcf.shape[0]
        pd1tcf_mm2 = pd1tcf_count / pd1tcf_area

        numbers_table = pd.DataFrame({
            'Sample': [sample_title],
            'CD8 area': [cd8_area],
            'CD8 count': [cd8_count],
            'CD8 per mm2': [cd8_mm2],
            'MHCII area': [mhcII_area],
            'MHCII count': [mhcII_count],
            'MHCII per mm2': [mhcII_mm2],
            'CD4 area': [cd4_area],
            'CD4 count': [cd4_count],
            'CD4 per mm2': [cd4_mm2],
            'TCF area': [tcf_area],
            'TCF count': [tcf_count],
            'TCF per mm2': [tcf_mm2],
            'PD1 area': [pd1_area],
            'PD1 count': [pd1_count],
            'PD1 per mm2': [pd1_mm2],
            'PD1TCF area': [pd1tcf_area],
            'PD1TCF count': [pd1tcf_count],
            'PD1TCF per mm2': [pd1tcf_mm2]})
        
        numbers_table.to_csv(f"./numbers/{sample_title}_numbers.csv", index=False)

        displaytable = ax3.table(cellText=numbers_table.values, colLabels=numbers_table.columns, cellLoc = 'center', colWidths=[0.05]*len(numbers_table.columns), loc='center')
        displaytable.set_fontsize(10)  #not working as autosize
        # displaytable.

        ax3.axis('off')
        # ax3.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

        ax3.set_title('Numbers/Output Data', fontsize=10,color='red', loc='center')
        
        ############### general and show ####################################################################################################################
        fig.suptitle(sample_title, fontsize=20, color='red', fontweight='bold')  # Add title to the figure

        # Save the plot
        plot_png = f"plots/{sample_title}_300dpi.png"
        plt.savefig(plot_png, dpi=300, bbox_inches='tight') #1200 possible

        # plt.show()

        plt.close()

        exit()

        # counts
        counts = data['Predictions'].value_counts()
        print(counts)


