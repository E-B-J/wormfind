# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:46:52 2024

@author: ebjam
"""
import joblib
import pandas as pd
# Load all iou results

# Load Look for false positives/false negs at an image level, and associate things

iou_files = [r"D:\toronto_microscopy\ixmc\rnAI-rep3-dec6_Plate_2441\rnAI-rep3-dec6_Plate_2441\TimePoint_1\dapi\one_field\iou_results.joblib",
             r"D:\toronto_microscopy\ixmc\RNai-p2-nov14_Plate_2383\RNai-p2-nov14_Plate_2383\TimePoint_1\dapi\one_field\iou_results.joblib",
             r"D:\toronto_microscopy\ixmc\RNai-nov14_Plate_2382\RNai-nov14_Plate_2382\TimePoint_1\dapi\one_field\iou_results.joblib"]

ai_detection_files = [r"D:\toronto_microscopy\ixmc\rnAI-rep3-dec6_Plate_2441\rnAI-rep3-dec6_Plate_2441\TimePoint_1\dapi\one_field\Nov26_detections_dy96_and_fish3500.joblib",
                      r"D:\toronto_microscopy\ixmc\RNai-p2-nov14_Plate_2383\RNai-p2-nov14_Plate_2383\TimePoint_1\dapi\one_field\Nov26_detections_dy96_and_fish3500.joblib",
                      r"D:\toronto_microscopy\ixmc\RNai-nov14_Plate_2382\RNai-nov14_Plate_2382\TimePoint_1\dapi\one_field\Nov26_detections_dy96_and_fish3500.joblib"]


# Use IoU to associate the images fromt hese lists together. Also look for commonality between false positives to aid in filtering.

false_positives = []
true_positives = []
false_negatives = []
false_pos_df = pd.DataFrame()
true_pos_df = pd.DataFrame()
iou_limits_df = pd.DataFrame()
for i in range(len(iou_files)):
    iou_file = joblib.load(iou_files[i])
    false_positives.extend(iou_file['unmatched_ais'])
    true_positives.extend(list(pd.DataFrame(iou_file['intersections'])['ai_name'].unique()))
    ai_detections = pd.DataFrame(joblib.load(ai_detection_files[i]))
    ai_detections = ai_detections.loc[ai_detections['worm_prediction'] == 'good'].copy()
    ai_detections['rep'] = int(i + 1)
    false_pos_df = pd.concat([false_pos_df, ai_detections])
    
    iou_limits = pd.DataFrame(iou_file['intersections'])
    iou_limits_df = pd.concat([iou_limits_df, iou_limits])
false_pos_df['true_positive'] = 5

false_pos_df.loc[false_pos_df['lookup'].isin(false_positives), 'true_positive'] = 0

# Set 'true_positive' to 1 where 'lookup' is in true_positives
false_pos_df.loc[false_pos_df['lookup'].isin(true_positives), 'true_positive'] = 1

#%%
import seaborn as sns
sns.violinplot(y='area', x='truePositive', data = false_pos_df, cut=0)

#%%
small_guys = false_pos_df.loc[false_pos_df['area'] <= 3000].copy()

#%%
predicted_worms = false_pos_df.loc[false_pos_df['worm_prediction'] == 'good'].copy()
rel_worms = predicted_worms.loc[predicted_worms['true_positive'] != 5].copy()
rel_worms['fish:dy_ratio_10k'] = rel_worms['dy96_pixels_10k'] / rel_worms['fish_pixels_3p5k']
falsies = rel_worms.loc[rel_worms['true_positive'] == 0].copy()
truthies = rel_worms.loc[rel_worms['true_positive'] == 1].copy()

of_interest = ['area', 'perimeter', 'circularity', 'solidity', 'dy96_percent_10k', 'fish_percent_10k', 'true_positive', 'fish:dy_ratio_10k']
rel_worm_interest = rel_worms[of_interest].copy().reset_index()
rel_worm_interest=rel_worm_interest.drop(['index'], axis=1)
sns.pairplot(rel_worm_interest, hue='true_positive')

# What do we see from the pair plot?
'''
I can filter on circularity! Real worms are not circles, circularity above ~0.22 are false

I can filter on upper area size! 35000 is too big - look at max size of true detection and then add 20%

I can filter on perimeter? Above 2k?

I can filter on dy96_10k % - above 50% = false!

I can filter on fish percent 10k - aboce 85 = false (for non epidermal species)

I can for sure filter on a ratio of dy96 to fish percent!
'''
#%%
import numpy as np

print(len(rel_worms))
print(sum(rel_worms['true_positive']))
print(sum(rel_worms['true_positive']) / len(rel_worms))
filter_worms = rel_worms.loc[rel_worms['circularity'] <= 0.22].copy()
print(len(filter_worms))
print(sum(filter_worms['true_positive']))
print(sum(filter_worms['true_positive']) / len(filter_worms))
filter_worms = filter_worms.loc[filter_worms['dy96_percent_10k'] <= 50].copy().reset_index()
filter_worms=filter_worms.drop(['index'], axis=1)
print(len(filter_worms))
print(sum(filter_worms['true_positive']))
print(sum(filter_worms['true_positive']) / len(filter_worms))
filter_worms = filter_worms.loc[filter_worms['perimeter'] <= 2000].copy().reset_index()
filter_worms=filter_worms.drop(['index'], axis=1)
print(len(filter_worms))
print(sum(filter_worms['true_positive']))
print(sum(filter_worms['true_positive']) / len(filter_worms))


sns.pairplot(filter_worms[['area', 'perimeter', 'circularity', 'solidity', 'dy96_percent_10k', 'fish_percent_10k', 'true_positive', 'fish:dy_ratio_10k']], hue='true_positive')



#%%
# Lets interrogate some of the lower ious. Obviously the stuff below one percent is a miss.
import cv2
import matplotlib.pyplot as plt


def contours_tos_masks_cv2(ai_contour, fiji_contour):
    mask = np.zeros((4096, 4096, 3), dtype=np.uint8)
    fiji_mask = np.zeros((4096, 4096), dtype=np.uint8)
    ai_mask = np.zeros((4096, 4096), dtype=np.uint8)
    if ai_contour is not None:
        cv2.drawContours(ai_mask, [ai_contour], -1, 255, thickness=cv2.FILLED)
        mask[ai_mask == 255] = [0, 0, 255]  # Red color for AI contour
    if fiji_contour is not None:
        cv2.drawContours(fiji_mask, [fiji_contour], -1, 255, thickness=cv2.FILLED)
        mask[fiji_mask == 255] = [255, 0, 0]
    overlap_mask = np.logical_and(fiji_mask == 255, ai_mask == 255)
    mask[overlap_mask] = [255, 255, 255]  # White color for overlap area
    return(mask)

low_iou = iou_limits_df.loc[iou_limits_df['IoU'] <= 0.55].copy()

images_to_load = list(low_iou['image'].unique())

for image in images_to_load:
    rele_low = low_iou.loc[low_iou['image'] == image].copy()
    for _, row in rele_low.iterrows():    
        fiji_roi = row['roi_name']
        ai_roi = row['ai_name']
        iou = row['IoU']
        ai_contour = row['ai_contour']
        fiji_cont = np.array(eval(row['fiji_contour']), dtype=np.int32).reshape((-1, 1, 2))
        mask = contours_tos_masks_cv2(ai_contour, fiji_cont)
        plt.figure(figsize=(8, 8))
        plt.imshow(mask)
        plt.axis('off')  # Hide axes for a cleaner image
        plt.show()
    


#%%
def contours_to_masks_cv2(ai_contour, fiji_contour):
    # Initialize a blank mask (single channel for simplicity)
    mask = np.zeros((4096, 4096, 3), dtype=np.uint8)  # RGB mask for colored contours
    fiji_mask = np.zeros((4096, 4096), dtype=np.uint8)
    ai_mask = np.zeros((4096, 4096), dtype=np.uint8)

    # Draw AI contour in red on AI mask and on the main mask
    if ai_contour is not None and len(ai_contour) > 0:
        cv2.drawContours(ai_mask, [ai_contour], -1, 255, thickness=cv2.FILLED)
        mask[ai_mask == 255] = [0, 0, 255]  # Red color for AI contour in main mask

    # Draw Fiji contour in blue on Fiji mask and on the main mask
    if fiji_contour is not None and len(fiji_contour) > 0:
        cv2.drawContours(fiji_mask, [fiji_contour], -1, 255, thickness=cv2.FILLED)
        mask[fiji_mask == 255] = [255, 0, 0]  # Blue color for Fiji contour in main mask

    # Overlap area: Where both contours overlap, set color to white
    overlap_mask = np.logical_and(fiji_mask == 255, ai_mask == 255)
    mask[overlap_mask] = [255, 255, 255]  # White color for overlap area

    return mask

# Assuming 'iou_limits_df' is already available and structured correctly
low_iou = iou_limits_df.loc[iou_limits_df['IoU'] <= 0.55].copy()

images_to_load = list(low_iou['image'].unique())

for image in images_to_load:
    rele_low = low_iou.loc[low_iou['image'] == image].copy()
    for _, row in rele_low.iterrows():    
        fiji_roi = row['roi_name']
        ai_roi = row['ai_name']
        iou = row['IoU']
        
        # Load the contours for both the AI and Fiji ROIs (assume they are stored as strings or lists)
        ai_contour = np.array(row['ai_contour'].iloc[0])  # Assume AI contour is already a valid numpy array or list of points
        fiji_cont = np.array(eval(row['fiji_contour']), dtype=np.int32).reshape((-1, 1, 2))  # Convert string to array if needed
        
        # Generate the mask with contours
        mask = contours_to_masks_cv2(ai_contour, fiji_cont)
        
        # Plotting the mask using matplotlib
        plt.figure(figsize=(8, 8))
        
        plt.imshow(mask)
        plt.axis('off') 
        plt.text(60, 150, "IoU = " + str(round(iou, 3)), fontsize = 20, color = 'w') # Hide axes for a cleaner image
        plt.show()

#%%
numpy_ai_cont = np.array(ai_contour.iloc[0])

#%%

# I think a true positive has an IoU greater than 0.5

true_positive_post_filters = iou_limits_df.loc[iou_limits_df['IoU'] > 0.5].copy()
fig, ax = plt.subplots(figsize=(10,10))
sns.violinplot(y='IoU', data=true_positive_post_filters, cut=0, ax=ax, inner=None, zorder = 0)
sns.swarmplot(y='IoU', data=true_positive_post_filters, ax=ax, palette='dark', zorder = 1)
sns.pointplot(y='IoU', data=true_positive_post_filters, ax=ax,color='k', scale = 2, markers=['D'])
sns.despine()

#%%
import os

dapi_ones = ["D:/toronto_microscopy/ixmc/RNai-nov14_Plate_2382/RNai-nov14_Plate_2382/TimePoint_1/dapi/one_field/",
             "D:/toronto_microscopy/ixmc/RNai-p2-nov14_Plate_2383/RNai-p2-nov14_Plate_2383/TimePoint_1/dapi/one_field/",
             "D:/toronto_microscopy/ixmc/rnAI-rep3-dec6_Plate_2441/rnAI-rep3-dec6_Plate_2441/TimePoint_1/dapi/one_field/"]

manual_fiji_data = pd.DataFrame()

for d_o in dapi_ones:
    csvs = [q for q in os.listdir(d_o) if q.endswith("csv")]
    for csv in csvs:
        m_d = pd.read_csv(d_o + csv)
        m_d['image'] = csv
        manual_fiji_data = pd.concat([manual_fiji_data, m_d])
        

#%%


sns.swarmplot(manual_fiji_data)
#%%
# False positives... Eugh
true_positive_post_filters_names = list(true_positive_post_filters['ai_name'])

all_ai_detect = pd.DataFrame()
for q in ai_detection_files:
    all_ai_detect = pd.concat([all_ai_detect, pd.DataFrame(joblib.load(q))])
#%%
all_worms = all_ai_detect.loc[all_ai_detect['worm_prediction'] == 'good'].copy().reset_index()
all_worms['true_positive'] = 0
all_worms.loc[all_worms['lookup'].isin(true_positive_post_filters_names), 'true_positive'] = 1
#%%
numeric_columns_list = ['true_positive',
'area',
'perimeter',
'circularity',
'solidity',
'dy96_pixels_10k',
'dy96_pixels_5k',
'dy96_percent_10k',
'fish_pixels_10k',
'fish_pixels_3p5k',
'fish_pixels_5k',
'fish_percent_10k',
'area_to_perimeter',
'aspect_ratio']



sns.pairplot(data = all_worms[numeric_columns_list], hue = 'true_positive')

#%%
all_worms['true_positive'] = all_worms['true_positive'].astype(int)
all_worms['col'] = all_worms['image'].str[-15]

anno = ['B', 'C']

matched_worms = all_worms.loc[all_worms['col'].isin(anno)].copy()
#%%


sns.pairplot(data = matched_worms[numeric_columns_list], hue='true_positive')

#%%
print(sum(matched_worms['true_positive']))
print(len(matched_worms))

#%%

print(sum(matched_worms['true_positive']))
print(len(matched_worms))
circ_filter = matched_worms.loc[matched_worms['circularity'] < 0.2].copy()
circ_filter['true_positive'] = circ_filter['true_positive'].astype(int)
print(sum(circ_filter['true_positive']))
print(len(circ_filter))
area_filter = circ_filter.loc[circ_filter['area'] < 27500].copy()
print(sum(area_filter['true_positive']))
print(len(area_filter))
dy96_filter = area_filter.loc[area_filter['dy96_percent_10k'] < 55].copy()
print(sum(dy96_filter['true_positive']))
print(len(dy96_filter))
#%%
# calculate area here!!

def area_calc(contour_type_thing):
    contour = np.array(eval(contour_type_thing), dtype=np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(contour)
    return(area)
manual_fiji_data['area'] = manual_fiji_data['ROI Coordinates'].apply(area_calc)
#%%

cont = np.array(eval(manual_fiji_data.iloc[0]['ROI Coordinates']), dtype=np.int32).reshape((-1, 1, 2))
print(area_calc(cont))
#%%

manual_fiji_data['row'] = manual_fiji_data['image'].str[-27]
manual_fiji_data['col'] = manual_fiji_data['image'].str[-26:-24]

l4440 = manual_fiji_data.loc[manual_fiji_data['row'] == 'B'].copy()
cyc1 = manual_fiji_data.loc[manual_fiji_data['row'] == 'C'].copy()
#sns.swarmplot(y='area', x ='col' , data = l4440)
sns.swarmplot(y='area', x ='col' , data = cyc1
              )

#%%
sns.swarmplot(y='area', data = all_worms, hue='true_positive')
#%%
# Now I want to join area, fish% etc across ai and man
def get_manual_name(ai_name):
    row_match = iou_limits_df.loc[iou_limits_df['ai_name'] == ai_name]
    if len(row_match) > 0:    
        manual_name = list([row_match['image'].values[0], row_match['roi_name'].values[0]])
    elif len(row_match) < 1:
        manual_name = None
    return (manual_name)


matched_worm_comparison = matched_worms.loc[matched_worms['true_positive'] == 1].copy()
matched_worm_comparison['man_name'] = matched_worm_comparison['lookup'].apply(get_manual_name)

duplicates = matched_worm_comparison.loc[matched_worm_comparison['man_name'].duplicated(keep=False)] 


def search_manual_worm_area(man_name):
    image = man_name[0]
    roi = man_name[1]
    manual_match = manual_fiji_data.loc[(manual_fiji_data['Image Name'] == image) & (manual_fiji_data['ROI Name'] == roi)]
    area = manual_match['area'].values[0]
    return area
    

#Use ROI name to get manual area
# I fucked the nameing! Area is now the manual anno...


matched_worm_comparison['manual_area'] = matched_worm_comparison['man_name'].apply(search_manual_worm_area)



#%%
fig, ax = plt.subplots(figsize = (8, 8))
sns.scatterplot(x='manual_area', y='area', data=matched_worm_comparison, ax=ax)
ax.set_ylabel('AI worm area')
ax.set_xlabel('Manual worm area')
sns.despine()


#%%
from scipy import stats

slope, intercept, r, p, std_err = stats.linregress(matched_worm_comparison['area'], matched_worm_comparison['manual_area'])



rez = stats.ttest_rel(matched_worm_comparison['area'], matched_worm_comparison['manual_area'])


#%%
# Lovely! Now I should download the embryo predicitons too, and see how they line up
p1_worms_and_embryo = joblib.load(r"D:\toronto_microscopy\ixmc\RNai-p2-nov14_Plate_2383\rnAI_p1_worms_AND_embryos.joblib")
p1_worms_and_embryo['rep'] = 1
p2_worms_and_embryo = joblib.load(r"D:\toronto_microscopy\ixmc\RNai-p2-nov14_Plate_2383\rnAI_p2_worms_AND_embryos.joblib")
p2_worms_and_embryo['rep'] = 2
p3_worms_and_embryo = joblib.load(r"D:\toronto_microscopy\ixmc\rnAI-rep3-dec6_Plate_2441\rnAI-rep3-dec6_Plate_2441\TimePoint_1\dy96\one_field\rnAI_p3_worms_AND_embryos.joblib")
p3_worms_and_embryo['rep'] = 3

empty_vector_embryo_reps = pd.concat([p1_worms_and_embryo, p2_worms_and_embryo, p3_worms_and_embryo])



#%%
#now match these two together!
import re
data = pd.read_csv(r"D:\toronto_microscopy\ixmc\RNai_1_2.csv")

spore_max = 2000000
def gen_spore_dict(spore_max):
    spore_dict = {}
    for i in range(0, 11):
        plate_col = i+1
        spores = spore_max / 2**i
        spore_dict.update({plate_col: spores})
    spore_dict.update({12 : 0})
    return(spore_dict)
di = gen_spore_dict(spore_max)

data['#spore'] = data['col'].map(di)
data['spores_per_ul'] = round(data['#spore'] / 65, 2)
data['ROI'] = data['roi_name'].str.extract(r'(\d{4}-\d{4})')
#%%

matched_embo_data = empty_vector_embryo_reps.loc[empty_vector_embryo_reps['lookup'].isin(true_positive_post_filters_names)].copy()



def get_manual_name(ai_name):
    row_match = iou_limits_df.loc[iou_limits_df['ai_name'] == ai_name]
    if len(row_match) > 0:    
        manual_name = list([row_match['image'].values[0], row_match['roi_name'].values[0]])
    elif len(row_match) < 1:
        manual_name = None
    return (manual_name)

def search_man_embo(roi_name):
    manual_match = data.loc[data['ROI'] == roi_name]
    embo = manual_match['embryos'].values[0]
    return embo

matched_embo_data['man_name'] = matched_embo_data['lookup'].apply(get_manual_name)
matched_embo_data_l4440 = matched_embo_data.loc[matched_embo_data['line'] == 'l4440'].copy()
matched_embo_data_l4440['ROI_name'] = matched_embo_data_l4440['man_name'].apply(lambda x: x[1] if len(x) > 1 else None)
matched_embo_data_l4440['man_embo'] = matched_embo_data_l4440['ROI_name'].apply(search_man_embo)

#%%
#For matching, I need to see if ri names are duped...

rois = manual_fiji_data["ROI Name"]
print(manual_fiji_data[rois.isin(rois[rois.duplicated()])].sort_values("ROI Name"))


#%%
fig, ax = plt.subplots(figsize=(15, 6))
sns.scatterplot(x='man_embo', y='emb >= 0.6', data = matched_embo_data_l4440, hue = 'rep', style='rep', ax=ax)

sns.despine()

#%%
def gravidity_level(embryos):
    if embryos == 0:
        lev = 'Non-gravid'
    elif embryos < 6:
        lev = 'Low'
    elif embryos < 20:
        lev = 'mid'
    else:
        lev = 'high'
    return lev

matched_embo_data_l4440['man_grav_level'] = matched_embo_data_l4440['man_embo'].apply(gravidity_level)

matched_embo_data_l4440['ai_grav_level'] = matched_embo_data_l4440['emb >= 0.6'].apply(gravidity_level)


#%%
category_names = [', 'Category2', 'Category3', 'Category4']  # Modify with your actual category names

# Create the heatmap with category labels
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=category_names, yticklabels=category_names)

# Optionally, add labels and a title
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()
#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
# matched_embo_data_l4440['man_grav_level'] = actual labels
# matched_embo_data_l4440['ai_grav_level'] = predicted labels

# Create a contingency table (cross-tabulation)
contingency_table = pd.crosstab(matched_embo_data_l4440['man_grav_level'], matched_embo_data_l4440['ai_grav_level'], normalize='index')

# The `normalize='index'` will normalize the rows (i.e., actual labels) to percentages.

# Convert the table to a long format for easy plotting
contingency_table = contingency_table.reset_index().melt(id_vars='man_grav_level', var_name='Predicted', value_name='Percentage')

# Plotting the clustered bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='man_grav_level', y='Percentage', hue='Predicted',order = ['Non-gravid', 'Low', 'mid', 'high'], hue_order = ['Non-gravid', 'Low', 'mid', 'high'], data=contingency_table, palette='Blues')

# Adding labels and title
plt.xlabel('Actual Labels')
plt.ylabel('Percentage of Predicted Labels')
plt.title('Percentage of Predicted Labels for Each Actual Label')
plt.legend(title='Predicted Labels', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()

