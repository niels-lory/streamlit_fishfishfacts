#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:13:12 2023

@author: nlory
"""


import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import tensorflow_addons as tfa
from PIL import Image


class_names_species = ['A73egs-p', 'Cunwcb-y', 'Istiophorus platypterus', 'P1rozc-z', 'Pqv7dp-s', 'Acanthaluteres brownii', 'Acanthaluteres spilomelanurus', 'Acanthaluteres vittiger', 'Acanthistius cinctus', 'Acanthopagrus australis', 'Acanthopagrus berda', 'Acanthopagrus latus', 'Achoerodus gouldii', 'Achoerodus viridis', 'Acreichthys tomentosus', 'Aesopia cornuta', 'Aethaloperca rogaa', 'Alectis ciliaris', 'Alectis indica', 'Alepes kleinii', 'Aluterus monoceros', 'Aluterus scriptus', 'Amanses scopas', 'Anampses caeruleopunctatus', 'Anampses elegans', 'Anampses femininus', 'Anampses geographicus', 'Anampses lennardi', 'Anampses melanurus', 'Anampses meleagrides', 'Anampses neoguinaicus', 'Anampses twistii', 'Anodontostoma chacunda', 'Anyperodon leucogrammicus', 'Aphareus furca', 'Aphareus rutilans', 'Aprion virescens', 'Argyrops spinifer', 'Aseraggodes melanostictus', 'Atractoscion aequidens', 'Atule mate', 'Auxis rochei', 'Auxis thazard', 'Bathylagichthys greyae', 'Beryx decadactylus', 'Bodianus anthioides', 'Bodianus axillaris', 'Bodianus bilunulatus', 'Bodianus bimaculatus', 'Bodianus diana', 'Bodianus loxozonus', 'Bodianus mesothorax', 'Bodianus perditio', 'Bodianus unimaculatus', 'Bodianus vulpinus', 'Bothus mancus', 'Bothus myriaster', 'Bothus pantherinus', 'Brachaluteres jacksonianus', 'Brachirus orientalis', 'Caesioperca lepidopterus', 'Cantherhines dumerilii', 'Cantherhines fronticinctus', 'Cantherhines pardalis', 'Cantheschenia grandisquamis', 'Caprodon longimanus', 'Caprodon schlegelii', 'Carangoides caeruleopinnatus', 'Carangoides chrysophrys', 'Carangoides equula', 'Carangoides ferdau', 'Carangoides fulvoguttatus', 'Carangoides hedlandensis', 'Carangoides malabaricus', 'Carangoides orthogrammus', 'Carangoides plagiotaenia', 'Caranx ignobilis', 'Caranx lugubris', 'Caranx melampygus', 'Caranx sexfasciatus', 'Carcharhinus albimarginatus', 'Carcharhinus amblyrhynchos', 'Carcharhinus falciformis', 'Carcharhinus galapagensis', 'Carcharhinus limbatus', 'Carcharhinus melanopterus', 'Carcharhinus obscurus', 'Carcharhinus plumbeus', 'Carcharhinus sorrah', 'Centroberyx affinis', 'Centrogenys vaigiensis', 'Centroscymnus coelolepis', 'Cephalopholis argus', 'Cephalopholis boenak', 'Cephalopholis cyanostigma', 'Cephalopholis formosa', 'Cephalopholis igarashiensis', 'Cephalopholis leopardus', 'Cephalopholis microprion', 'Cephalopholis miniata', 'Cephalopholis sexmaculata', 'Cephalopholis sonnerati', 'Cephalopholis spiloparaea', 'Chascanopsetta lugubris', 'Cheilinus chlorourus', 'Cheilinus fasciatus', 'Cheilinus oxycephalus', 'Cheilinus trilobatus', 'Cheilinus undulatus', 'Cheilio inermis', 'Cheilodactylus ephippium', 'Cheilodactylus fuscus', 'Cheilodactylus spectabilis', 'Cheilodactylus vestitus', 'Chelidonichthys kumu', 'Chirocentrus dorab', 'Chirocentrus nudus', 'Choerodon anchorago', 'Choerodon cauteroma', 'Choerodon cyanodus', 'Choerodon fasciatus', 'Choerodon graphicus', 'Choerodon jordani', 'Choerodon rubescens', 'Choerodon schoenleinii', 'Choerodon venustus', 'Choerodon vitta', 'Choerodon zamboangae', 'Chromileptes altivelis', 'Cirrhilabrus bathyphilus', 'Cirrhilabrus condei', 'Cirrhilabrus cyanopleura', 'Cirrhilabrus exquisitus', 'Cirrhilabrus laboutei', 'Cirrhilabrus punctatus', 'Cirrhilabrus scottorum', 'Cirrhilabrus temminckii', 'Coris aygula', 'Coris batuensis', 'Coris bulbifrons', 'Coris caudimacula', 'Coris dorsomacula', 'Coris gaimard', 'Coris picta', 'Coris pictoides', 'Coris sandeyeri', 'Crenimugil crenilabis', 'Cymbacephalus nematophthalmus', 'Cymolutes praetextatus', 'Cymolutes torquatus', 'Cynoglossus puncticeps', 'Cyttopsis rosea', 'Dactylophora nigricans', 'Decapterus macrosoma', 'Decapterus russelli', 'Diproctacanthus xanthurus', 'Dotalabrus aurantiacus', 'Elagatis bipinnulata', 'Epibulus insidiator', 'Epinephelus areolatus', 'Epinephelus bleekeri', 'Epinephelus chlorostigma', 'Epinephelus coeruleopunctatus', 'Epinephelus coioides', 'Epinephelus corallicola', 'Epinephelus cyanopodus', 'Epinephelus epistictus', 'Epinephelus fasciatus', 'Epinephelus fuscoguttatus', 'Epinephelus hexagonatus', 'Epinephelus howlandi', 'Epinephelus lanceolatus', 'Epinephelus latifasciatus', 'Epinephelus macrospilos', 'Epinephelus maculatus', 'Epinephelus melanostigma', 'Epinephelus merra', 'Epinephelus morrhua', 'Epinephelus multinotatus', 'Epinephelus ongus', 'Epinephelus polyphekadion', 'Epinephelus quoyanus', 'Epinephelus radiatus', 'Epinephelus retouti', 'Epinephelus rivulatus', 'Epinephelus sexfasciatus', 'Epinephelus spilotoceps', 'Epinephelus tauvina', 'Epinephelus undulatostriatus', 'Etelis carbunculus', 'Etelis coruscans', 'Eubalichthys cyanoura', 'Eubalichthys mosaicus', 'Eupetrichthys angustipes', 'Euthynnus affinis', 'Evistias acutirostris', 'Gempylus serpens', 'Gnathanodon speciosus', 'Gnathodentex aureolineatus', 'Gracila albomarginata', 'Gymnocranius audleyi', 'Gymnocranius euanus', 'Gymnocranius grandoculis', 'Gymnocranius microdon', 'Gymnosarda unicolor', 'Halichoeres argus', 'Halichoeres biocellatus', 'Halichoeres chloropterus', 'Halichoeres chrysus', 'Halichoeres hartzfeldii', 'Halichoeres hortulanus', 'Halichoeres leucurus', 'Halichoeres margaritaceus', 'Halichoeres marginatus', 'Halichoeres melanochir', 'Halichoeres melanurus', 'Halichoeres melasmapomus', 'Halichoeres miniatus', 'Halichoeres nebulosus', 'Halichoeres nigrescens', 'Halichoeres scapularis', 'Halichoeres trimaculatus', 'Harriotta raleighana', 'Hemigymnus fasciatus', 'Hemigymnus melapterus', 'Hemiramphus far', 'Herklotsichthys quadrimaculatus', 'Hologymnosus annulatus', 'Hologymnosus doliatus', 'Hyporhamphus affinis', 'Hyporhamphus dussumieri', 'Inegocia japonica', 'Johnius borneensis', 'Katsuwonus pelamis', 'Labrichthys unilineatus', 'Labroides bicolor', 'Labroides dimidiatus', 'Labroides pectoralis', 'Labropsis australis', 'Labropsis manabei', 'Labropsis xanthonota', 'Latridopsis forsteri', 'Lepidocybium flavobrunneum', 'Leptojulis cyanopleura', 'Lethrinus amboinensis', 'Lethrinus atkinsoni', 'Lethrinus erythracanthus', 'Lethrinus genivittatus', 'Lethrinus harak', 'Lethrinus lentjan', 'Lethrinus microdon', 'Lethrinus miniatus', 'Lethrinus nebulosus', 'Lethrinus obsoletus', 'Lethrinus olivaceus', 'Lethrinus ornatus', 'Lethrinus rubrioperculatus', 'Lethrinus semicinctus', 'Lethrinus variegatus', 'Lethrinus xanthochilus', 'Liopropoma mitratum', 'Liopropoma susumi', 'Liza subviridis', 'Liza vaigiensis', 'Lniistius aneitensis', 'Lniistius pavo', 'Lutjanus adetii', 'Lutjanus argentimaculatus', 'Lutjanus biguttatus', 'Lutjanus bohar', 'Lutjanus carponotatus', 'Lutjanus decussatus', 'Lutjanus ehrenbergii', 'Lutjanus erythropterus', 'Lutjanus fulviflamma', 'Lutjanus fulvus', 'Lutjanus gibbus', 'Lutjanus johnii', 'Lutjanus kasmira', 'Lutjanus lemniscatus', 'Lutjanus lutjanus', 'Lutjanus malabaricus', 'Lutjanus monostigma', 'Lutjanus quinquelineatus', 'Lutjanus rivulatus', 'Lutjanus russellii', 'Lutjanus sebae', 'Lutjanus semicinctus', 'Lutjanus semicinctus quoy', 'Lutjanus timoriensis', 'Lutjanus vitta', 'Macolor macularis', 'Macolor niger', 'Macropharyngodon choati', 'Macropharyngodon kuiteri', 'Macropharyngodon meleagris', 'Macropharyngodon negrosensis', 'Macropharyngodon ornatus', 'Megalaspis cordyla', 'Meuschenia australis', 'Meuschenia freycineti', 'Meuschenia galii', 'Meuschenia hippocrepis', 'Meuschenia scaber', 'Meuschenia trachylepis', 'Monacanthus chinensis', 'Monotaxis grandoculis', 'Mugim cephalus', 'Naucrates ductor', 'Negaprion acutidens', 'Nemadactylus douglasii', 'Nemipterus furcosus', 'Nemipterus hexodon', 'Nemipterus peronii', 'Netuma thalassina', 'Nibea soldado', 'Notolabrus fucicola', 'Notolabrus gymnogenis', 'Notolabrus tetricus', 'Notorynchus cepedianus', 'Novaculichthys taeniourus', 'Novaculoides macrolepidotus', 'Oedalechilus labiosus', 'Ophthalmolepis lineolatus', 'Otolithes ruber', 'Oxycheilinus bimaculatus', 'Oxycheilinus celebicus', 'Oxycheilinus digrammus', 'Oxycheilinus unifasciatus', 'Oxymonacanthus longirostris', 'Pagrus auratus', 'Paracaesio kusakarii', 'Paracheilinus filamentosus', 'Paraluteres prionurus', 'Paramonacanthus choirocephalus', 'Paraplagusia bilineata', 'Parastromateus niger', 'Pardachirus hedleyi', 'Pardachirus pavoninus', 'Pentapodus aureofasciatus', 'Pentapodus paradiseus', 'Pentapodus vitta quoy', 'Pervagor alternans', 'Pervagor aspricaudus', 'Pervagor janthinosoma', 'Pervagor melanocephalus', 'Pervagor nigrolineatus', 'Pinjalo lewisi', 'Platycephalus indicus', 'Plectranthias longimanus', 'Plectranthias nanus', 'Plectranthias winniensis', 'Plectropomus areolatus', 'Plectropomus laevis', 'Plectropomus leopardus', 'Plectropomus maculatus', 'Plectropomus oligacanthus', 'Plotosus lineatus', 'Pristipomoides argyrogrammicus', 'Pristipomoides auricilla', 'Pristipomoides filamentosus', 'Pristipomoides flavipinnis', 'Pristipomoides sieboldii', 'Pristipomoides zonatus', 'Promethichthys prometheus', 'Protonibea diacanthus', 'Psettodes erumei', 'Pseudalutarius nasicornis', 'Pseudanthias bicolor', 'Pseudanthias cooperi', 'Pseudanthias dispar', 'Pseudanthias fasciatus', 'Pseudanthias huchtii', 'Pseudanthias hypselosoma', 'Pseudanthias lori', 'Pseudanthias luzonensis', 'Pseudanthias pictilis', 'Pseudanthias pleurotaenia', 'Pseudanthias rubrizonatus', 'Pseudanthias sheni', 'Pseudanthias smithvanizi', 'Pseudanthias squamipinnis', 'Pseudanthias tuka', 'Pseudanthias ventralis', 'Pseudocaranx dentex', 'Pseudocarcharias kamoharai', 'Pseudocheilinus evanidus', 'Pseudocheilinus hexataenia', 'Pseudocheilinus ocellatus', 'Pseudocheilinus octotaenia', 'Pseudodax moluccanus', 'Pseudojuloides cerasinus', 'Pseudolabrus biserialis', 'Pseudolabrus guentheri', 'Pseudolabrus luculentus', 'Pseudorhombus argus', 'Pseudorhombus arsius', 'Pseudorhombus elevatus', 'Pteragogus cryptus', 'Pteragogus enneacanthus', 'Pteragogus flagellifer', 'Rastrelliger kanagurta', 'Retropinna semoni', 'Rhabdosargus sarba', 'Rhincodon typus', 'Rhizoprionodon acutus', 'Ruvettus pretiosus', 'Samaris cristatus', 'Samariscus triocellatus', 'Sarda orientalis', 'Sardinella albella', 'Sardinella gibbosa', 'Sardinops sagax', 'Scaevius milii', 'Scolopsis affinis', 'Scolopsis bilineata', 'Scolopsis lineata', 'Scolopsis margaritifer', 'Scolopsis monogramma', 'Scolopsis trilineata', 'Scolopsis vosmeri', 'Scolopsis xenochrous', 'Scomberoides commersonnianus', 'Scomberoides lysan', 'Scomberomorus commerson', 'Selar crumenophthalmus', 'Selaroides leptolepis', 'Seriola dumerili', 'Seriola hippos', 'Seriola rivoliana', 'Seriolina nigrofasciata', 'Serranocirrhitus latus', 'Sillago ciliata', 'Sillago sihama', 'Soleichthys heterorhinos', 'Sphyraena barracuda', 'Sphyraena forsteri', 'Sphyraena jello', 'Sphyraena obtusata', 'Stegostoma fasciatum', 'Stethojulis bandanensis', 'Stethojulis interrupta', 'Stethojulis strigiventer', 'Stethojulis trilineata', 'Stolephorus waitei', 'Suezichthys arquatus', 'Suezichthys cyanolaemus', 'Suezichthys gracilis', 'Symphorichthys spilurus', 'Symphorus nematophorus', 'Thalassoma amblycephalum', 'Thalassoma hardwicke', 'Thalassoma jansenii', 'Thalassoma lunare', 'Thalassoma lutescens', 'Thalassoma nigrofasciatum', 'Thalassoma purpureum', 'Thalassoma quinquevittatum', 'Thalassoma trilobatum', 'Thryssa baelama', 'Thryssa hamiltonii', 'Thunnus alalunga', 'Thunnus albacares', 'Thysanophrys celebica', 'Thysanophrys chiltonae', 'Trachichthys australis', 'Trachinotus baillonii', 'Trachinotus blochii', 'Trachinotus botla', 'Trachypoma macracanthus', 'Triaenodon obesus', 'Uraspis secunda', 'Valamugil cunnesius', 'Valamugil engeli', 'Valamugil seheli', 'Variola albimarginata', 'Variola louti', 'Wattsia mossambica', 'Wetmorella albofasciata', 'Wetmorella nigropinnata', 'Xiphocheilus typus', 'Zenarchopterus dispar', 'Zeus faber']
class_names_genus = ['A73Egs-P', 'Cunwcb-Y', 'Istiophorus', 'P1Rozc-Z', 'Pqv7Dp-S', 'Acanthaluteres', 'Acanthistius', 'Acanthopagrus', 'Achoerodus', 'Acreichthys', 'Aesopia', 'Aethaloperca', 'Alectis', 'Alepes', 'Aluterus', 'Amanses', 'Anampses', 'Anodontostoma', 'Anyperodon', 'Aphareus', 'Aprion', 'Argyrops', 'Aseraggodes', 'Atractoscion', 'Atule', 'Auxis', 'Bathylagichthys', 'Beryx', 'Bodianus', 'Bothus', 'Brachaluteres', 'Brachirus', 'Caesioperca', 'Cantherhines', 'Cantheschenia', 'Caprodon', 'Carangoides', 'Caranx', 'Carcharhinus', 'Centroberyx', 'Centrogenys', 'Centroscymnus', 'Cephalopholis', 'Chascanopsetta', 'Cheilinus', 'Cheilio', 'Cheilodactylus', 'Chelidonichthys', 'Chirocentrus', 'Choerodon', 'Chromileptes', 'Cirrhilabrus', 'Coris', 'Crenimugil', 'Cymbacephalus', 'Cymolutes', 'Cynoglossus', 'Cyttopsis', 'Dactylophora', 'Decapterus', 'Diproctacanthus', 'Dotalabrus', 'Elagatis', 'Epibulus', 'Epinephelus', 'Etelis', 'Eubalichthys', 'Eupetrichthys', 'Euthynnus', 'Evistias', 'Gempylus', 'Gnathanodon', 'Gnathodentex', 'Gracila', 'Gymnocranius', 'Gymnosarda', 'Halichoeres', 'Harriotta', 'Hemigymnus', 'Hemiramphus', 'Herklotsichthys', 'Hologymnosus', 'Hyporhamphus', 'Inegocia', 'Johnius', 'Katsuwonus', 'Labrichthys', 'Labroides', 'Labropsis', 'Latridopsis', 'Lepidocybium', 'Leptojulis', 'Lethrinus', 'Liopropoma', 'Liza', 'Lniistius', 'Lutjanus', 'Macolor', 'Macropharyngodon', 'Megalaspis', 'Meuschenia', 'Monacanthus', 'Monotaxis', 'Mugim', 'Naucrates', 'Negaprion', 'Nemadactylus', 'Nemipterus', 'Netuma', 'Nibea', 'Notolabrus', 'Notorynchus', 'Novaculichthys', 'Novaculoides', 'Oedalechilus', 'Ophthalmolepis', 'Otolithes', 'Oxycheilinus', 'Oxymonacanthus', 'Pagrus', 'Paracaesio', 'Paracheilinus', 'Paraluteres', 'Paramonacanthus', 'Paraplagusia', 'Parastromateus', 'Pardachirus', 'Pentapodus', 'Pervagor', 'Pinjalo', 'Platycephalus', 'Plectranthias', 'Plectropomus', 'Plotosus', 'Pristipomoides', 'Promethichthys', 'Protonibea', 'Psettodes', 'Pseudalutarius', 'Pseudanthias', 'Pseudocaranx', 'Pseudocarcharias', 'Pseudocheilinus', 'Pseudodax', 'Pseudojuloides', 'Pseudolabrus', 'Pseudorhombus', 'Pteragogus', 'Rastrelliger', 'Retropinna', 'Rhabdosargus', 'Rhincodon', 'Rhizoprionodon', 'Ruvettus', 'Samaris', 'Samariscus', 'Sarda', 'Sardinella', 'Sardinops', 'Scaevius', 'Scolopsis', 'Scomberoides', 'Scomberomorus', 'Selar', 'Selaroides', 'Seriola', 'Seriolina', 'Serranocirrhitus', 'Sillago', 'Soleichthys', 'Sphyraena', 'Stegostoma', 'Stethojulis', 'Stolephorus', 'Suezichthys', 'Symphorichthys', 'Symphorus', 'Thalassoma', 'Thryssa', 'Thunnus', 'Thysanophrys', 'Trachichthys', 'Trachinotus', 'Trachypoma', 'Triaenodon', 'Uraspis', 'Valamugil', 'Variola', 'Wattsia', 'Wetmorella', 'Xiphocheilus', 'Zenarchopterus', 'Zeus']

def translate_class_index(class_index, class_names):
    class_dict = {key: idx for idx, key in enumerate(class_names)}
    key_list = list(class_dict.keys())
    val_list = list(class_dict.values())
    # print key
    position = val_list.index(class_index)
    return key_list[position]
 
def find_maxima(predictions, class_names):
    sorted_pred= np.sort(predictions)
    a = sorted_pred[0][-2:-1]
    a = list(predictions[0]).index(a)
    b = sorted_pred[0][-3:-2]
    b = list(predictions[0]).index(b)
    c = sorted_pred[0][-4:-3]
    c = list(predictions[0]).index(c)
    return translate_class_index(a,class_names), translate_class_index(b,class_names), translate_class_index(c,class_names)


def classify(model_folder, image_input, image_dim, class_names):
    model_dir = model_folder
    model = keras.models.load_model(model_dir)
    # Convert the image to a numpy array
    
    #img_array = image.img_to_array(image_input)
    #img_array = np.array(image_input)
    img_array = img
    
    # Expand the shape of the image array from (224, 224, 3) to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values from [0, 255] to [-1, 1]
    img_array = (img_array / 127.5) - 1
    # Use the model to make a prediction
    predictions = model.predict(img_array)
    # Get the class with the highest prediction probability
    class_index = np.argmax(predictions[0])
    # next most likely predictions:
    pred_1 = translate_class_index(class_index, class_names)
    pred_2,pred_3, pred_4 = find_maxima(predictions,class_names)
    # Print the predicted class
    return pred_1, pred_2, pred_3, pred_4
    
def fishfacts(species, column):
    fish_facts_df = pd.read_csv("data/fish_facts.csv")
    fish_fact_query = fish_facts_df[fish_facts_df['class_names_transformed'] == species][column]
    # workaround to get JUST the entry from the pandas DataFrame
    fish_fact_query = list(fish_fact_query)
    fish_fact_query = fish_fact_query[0]
    return fish_fact_query


#st.title('fish fish facts')

#label = "Upload your image here!"

uploaded_file = st.file_uploader(label="", type=['png', 'jpg'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if uploaded_file:  # if user uploaded file
        #################################### Predictions ####################################
        
        # uploaded image for display            
        img_display = Image.open(uploaded_file)
        # load IUCN red list legend
        iucn_legend = Image.open('data/red_list_legend.png')
        # tranform image dimensions for prediction
        img = image.img_to_array(img_display)        
        img = tf.image.resize(img, [224,224], preserve_aspect_ratio=False)
        # predict species:
        pred_1, pred_2, pred_3, pred_4 = classify("models/230203_modelmaker_224px", img, 224, class_names_species)
        # predict genus:
        pred_5, pred_6, pred_7, pred_8 = classify("models/familymodel", img, 224, class_names_genus)
        # loading fish facts
        conservation_status = fishfacts(pred_1, "category")
        main_common_name = fishfacts(pred_1,"main_common_name")
                
        #################################### Output ####################################
        
        # Prediction
        col1, col2 = st.columns(2)

        with col1:
            st.header(f':blue[Predicted species] :fish:')
            st.subheader(f"1. _{pred_1}_ \n2. _{pred_2}_ \n3. _{pred_3}_ \n4. _{pred_4}_")
            st.caption(f"You think your fish was not among the predictions? Maybe you're more lucky searching for these genus names: {pred_5}, {pred_6}, {pred_7} or {pred_8}")

        with col2:
            st.header('')
            # display the uploaded picture
            st.image(img_display, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            st.caption(f'Predicted as _{pred_1}_, common name {main_common_name} ')
        
        # Conservation status
        col1, col2 = st.columns(2)
        
        with col1:
            st.header(f':blue[Conservation status]')
            st.subheader(f"{conservation_status} \nIUCN red list of species: https://www.iucnredlist.org/ ")  
        
        with col2:
            st.header('')
            st.image(iucn_legend, caption='')
        
        #################################### IUCN text ####################################
        
        # rationale
        with st.container():    
            st.header(f':fish: :blue[Rationale] :fish:')
            st.caption('https://www.iucnredlist.org/')
            rationale = fishfacts(pred_1, "rationale")
            st.caption(rationale)
        
        # geographicrange
        with st.container():    
            st.header(f':fish: :blue[Geographic range] :fish:')
            st.caption('https://www.iucnredlist.org/')
            geographicrange = fishfacts(pred_1, "geographicrange")
            st.caption(geographicrange)

        # population
        with st.container():    
            st.header(f':fish: :blue[Population] :fish:')
            st.caption('https://www.iucnredlist.org/')
            population = fishfacts(pred_1, "population")
            st.caption(population)

        # habitat
        with st.container():    
            st.header(f':fish: :blue[Habitat] :fish:')
            st.caption('https://www.iucnredlist.org/')
            habitat = fishfacts(pred_1, "habitat")
            st.caption(habitat)
            
        # threats
        with st.container():    
            st.header(f':fish: :blue[Threats] :fish:')
            st.caption('https://www.iucnredlist.org/')
            threats = fishfacts(pred_1, "threats")
            st.caption(threats)

        # conservationmeasures
        with st.container():    
            st.header(f':fish: :blue[Conservation measures] :fish:')
            st.caption('https://www.iucnredlist.org/')
            conservationmeasures = fishfacts(pred_1, "conservationmeasures")
            st.caption(conservationmeasures)

        # usetrade
        with st.container():    
            st.header(f':fish: :blue[Usetrade] :fish:')
            st.caption('https://www.iucnredlist.org/')
            usetrade = fishfacts(pred_1, "usetrade")
            st.caption(usetrade)
            
         # Reference
        with st.container():    
            st.header(f':fish: :red[Reference] :fish:')
            st.caption('IUCN. 2022. The IUCN Red List of Threatened Species. Version 2022-2. https://www.iucnredlist.org. Accessed on [01.02.2023].' )
             
