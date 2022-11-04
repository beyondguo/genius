def get_label2desc(dataset_name):
    if 'ng' in dataset_name:
        label2desc = {0:"alt atheism",
                1:"computer graphics",
                2:"computer os microsoft windows misc",
                3:"computer system ibm pc hardware",
                4:"computer system mac hardware",
                5:"computer windows x",
                6:"misc for sale",
                7:"rec autos auto",
                8:"rec motorcycles",
                9:"rec sport baseball",
                10:"rec sport hockey",
                11:"sci crypt",
                12:"sci electronics",
                13:"sci medicine med",
                14:"sci space universe",
                15:"soc religion christian",
                16:"talk politics guns gun",
                17:"talk politics mideast",
                18:"talk politics misc",
                19:"talk religion misc"}
    elif 'bbc' in dataset_name or '5huffpost' in dataset_name:
        label2desc = None
    elif 'yahoo' in dataset_name:
        label2desc = {0: "Society Culture",
                    1: "Science Mathematics",
                    2: "Health",
                    3: "Education Reference",
                    4: "Computers Internet",
                    5: "Sports",
                    6: "Business Finance",
                    7: "Entertainment Music",
                    8: "Family Relationships",
                    9: "Politics Government"}
    elif 'imdb' in dataset_name or 'sst2' in dataset_name:  
        label2desc = {0: "negative, bad", 1: "positive, good"}
    else:
        print(f"{dataset_name} not supported! Please add the label info into `baselines/label_desc.py`")
    return label2desc