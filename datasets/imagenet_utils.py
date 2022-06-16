# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json

# with open("./imagenet2synset.json", 'r') as f:
#     imagenet_to_synset = json.load(f)

# with open("./imagenetclass992.json", 'r') as f:
#     synset_to_imagenet = json.load(f)

pascal_to_synset = {
	"bird": ["cock (n01514668)", "hen (n01514859)", "water ouzel (n01601694)", "house finch (n01532829)", "brambling (n01530575)", "junco (n01534433)", "goldfinch (n01531178)", "indigo bunting (n01537544)", "chickadee (n01592084)", "robin (n01558993)", "bulbul (n01560419)", "magpie (n01582220)", "jay (n01580077)", "black swan (n01860187)", "crane (n02012849)", "spoonbill (n02006656)", "flamingo (n02007558)", "bustard (n02018795)", "limpkin (n02013706)", "bittern (n02011460)", "little blue heron (n02009229)", "American egret (n02009912)", "oystercatcher (n02037110)", "dowitcher (n02033041)", "red-backed sandpiper (n02027492)", "redshank (n02028035)", "ruddy turnstone (n02025239)", "black stork (n02002724)", "white stork (n02002556)", "American coot (n02018207)", "king penguin (n02056570)", "albatross (n02058221)", "pelican (n02051845)", "European gallinule (n02017213)", "goose (n01855672)", "drake (n01847000)", "red-breasted merganser (n01855032)", "coucal (n01824575)", "hummingbird (n01833805)", "ostrich (n01518878)", "ruffed grouse (n01797886)", "black grouse (n01795545)", "prairie chicken (n01798484)", "ptarmigan (n01796340)", "quail (n01806567)", "partridge (n01807496)", "peacock (n01806143)", "bee eater (n01828970)", "hornbill (n01829413)", "vulture (n01616318)", "great grey owl (n01622779)", "kite (n01608432)", "bald eagle (n01614925)", "jacamar (n01843065)", "toucan (n01843383)", "macaw (n01818515)", "African grey (n01817953)", "lorikeet (n01820546)", "sulphur-crested cockatoo (n01819313)"],
    "boat": ["gondola (n03447447)", "fireboat (n03344393)", "yawl (n04612504)", "canoe (n02951358)", "lifeboat (n03662601)", "speedboat (n04273569)"],
    "bottle": ["beer bottle (n02823428)", "wine bottle (n04591713)", "water bottle (n04557648)", "pop bottle (n03983396)", "pill bottle (n03937543)", "whiskey jug (n04579145)", "water jug (n04560804)"],
    "bus": ["minibus (n03769881)", "school bus (n04146614)", "trolleybus (n04487081)"],
    "car": ["ambulance (n02701002)", "limousine (n03670208)", "jeep (n03594945)", "Model T (n03777568)", "cab (n02930766)", "minivan (n03770679)", "convertible (n03100240)", "racer (n04037443)", "beach wagon (n02814533)", "sports car (n04285008)"],
    "cat": ["Egyptian cat (n02124075)", "Persian cat (n02123394)", "tabby (n02123045)", "Siamese cat (n02123597)", "tiger cat (n02123159)", "cougar (n02125311)", "lynx (n02127052)"],
    "chair": ["barber chair (n02791124)", "rocking chair (n04099969)", "folding chair (n03376595)", "throne (n04429376)"],
    "diningtable": ["dining table (n03201208)"],
    "dog": ["dalmatian (n02110341)", "Mexican hairless (n02113978)", "pug (n02110958)", "Newfoundland (n02111277)", "Leonberg (n02111129)", "basenji (n02110806)", "Great Pyrenees (n02111500)", "Eskimo dog (n02109961)", "bull mastiff (n02108422)", "Saint Bernard (n02109525)", "Great Dane (n02109047)", "boxer (n02108089)", "Rottweiler (n02106550)", "Old English sheepdog (n02105641)", "Shetland sheepdog (n02105855)", "kelpie (n02105412)", "Border collie (n02106166)", "Bouvier des Flandres (n02106382)", "German shepherd (n02106662)", "komondor (n02105505)", "briard (n02105251)", "collie (n02106030)", "groenendael (n02105056)", "malinois (n02105162)", "French bulldog (n02108915)", "kuvasz (n02104029)", "schipperke (n02104365)", "Doberman (n02107142)", "affenpinscher (n02110627)", "miniature pinscher (n02107312)", "Tibetan mastiff (n02108551)", "Siberian husky (n02110185)", "malamute (n02110063)", "Bernese mountain dog (n02107683)", "Appenzeller (n02107908)", "EntleBucher (n02108000)", "Greater Swiss Mountain dog (n02107574)", "toy poodle (n02113624)", "miniature poodle (n02113712)", "standard poodle (n02113799)", "Pembroke (n02113023)", "Cardigan (n02113186)", "Rhodesian ridgeback (n02087394)", "Scottish deerhound (n02092002)", "bloodhound (n02088466)", "otterhound (n02091635)", "Afghan hound (n02088094)", "redbone (n02090379)", "bluetick (n02088632)", "basset (n02088238)", "Ibizan hound (n02091244)", "Saluki (n02091831)", "Norwegian elkhound (n02091467)", "beagle (n02088364)", "Weimaraner (n02092339)", "black-and-tan coonhound (n02089078)", "borzoi (n02090622)", "Irish wolfhound (n02090721)", "English foxhound (n02089973)", "Walker hound (n02089867)", "whippet (n02091134)", "Italian greyhound (n02091032)", "Dandie Dinmont (n02096437)", "Norwich terrier (n02094258)", "Border terrier (n02093754)", "West Highland white terrier (n02098286)", "Yorkshire terrier (n02094433)", "Airedale (n02096051)", "Irish terrier (n02093991)", "Bedlington terrier (n02093647)", "Norfolk terrier (n02094114)", "Lhasa (n02098413)", "silky terrier (n02097658)", "Kerry blue terrier (n02093859)", "Scotch terrier (n02097298)", "Tibetan terrier (n02097474)", "cairn (n02096177)", "soft-coated wheaten terrier (n02098105)", "Boston bull (n02096585)", "Australian terrier (n02096294)", "Staffordshire bullterrier (n02093256)", "American Staffordshire terrier (n02093428)", "Lakeland terrier (n02095570)", "Sealyham terrier (n02095889)", "giant schnauzer (n02097130)", "miniature schnauzer (n02097047)", "standard schnauzer (n02097209)", "wire-haired fox terrier (n02095314)", "curly-coated retriever (n02099429)", "flat-coated retriever (n02099267)", "golden retriever (n02099601)", "Chesapeake Bay retriever (n02099849)", "Labrador retriever (n02099712)", "Sussex spaniel (n02102480)", "Brittany spaniel (n02101388)", "clumber (n02101556)", "cocker spaniel (n02102318)", "English springer (n02102040)", "Welsh springer spaniel (n02102177)", "Irish water spaniel (n02102973)", "vizsla (n02100583)", "German short-haired pointer (n02100236)", "Irish setter (n02100877)", "Gordon setter (n02101006)", "English setter (n02100735)", "Maltese dog (n02085936)", "Chihuahua (n02085620)", "Pekinese (n02086079)", "Shih-Tzu (n02086240)", "toy terrier (n02087046)", "Japanese spaniel (n02085782)", "papillon (n02086910)", "Blenheim spaniel (n02086646)", "Brabancon griffon (n02112706)", "Samoyed (n02111889)", "Pomeranian (n02112018)", "keeshond (n02112350)", "chow (n02112137)"],
	"horse": ["sorrel (n02389026)"],
	#"motorbike": ["moped (n03785016)"],
	"person": ["ballplayer (n09835506)", "ballplayer (n09835506)", "groom (n10148035)", "scuba diver (n10565667)"],
	#"pottedplant": ["daisy (n11939491)", "yellow lady slipper (n12057211)"],
	"sheep": ["ram (n02412080)"],
	#"sofa": ["studio couch (n04344873)"],
	"train": ["bullet train (n02917067)"],
	#"tvmonitor": ["monitor (n03782006)"],
	"aeroplane": ["airliner (n02690373)"],
	"bicycle": ["bicycle-built-for-two (n02835271)", "mountain bike (n03792782)"],
}

def parse_synset_str(x):
    synset = ''
    i, j = 0, 0
    while True:
        if x[i] == '(':
            j = i + 1
            while True:
                if x[j] == ')':
                    break
                synset += x[j]
                j += 1
            
            break
        i += 1

    return synset


    
pascal_to_id = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    #'motorbike': 13,
    'person': 13, 
    #'pottedplant': 15,
    'sheep': 14,
    #'sofa': 17,
    'train': 15,
    #'tvmonitor': 19,
}

pascal_to_coco = {
    'background': 0,
    'aeroplane': 5,
    'bicycle': 2,
    'bird': 16,
    'boat': 9,
    'bottle': 44,
    'bus': 6,
    'car': 3,
    'cat': 17,
    'chair': 62,
    #'cow': 21,
    'diningtable': 67,
    'dog': 18,
    'horse': 19,
    #'motorbike': 4,
    'person': 1, 
    #'pottedplant': 64,
    'sheep': 20,
    #'sofa': 63,
    'train': 7,
    #'tvmonitor': 72,
}

random_synset_100 = ['n02966193', 'n02825657', 'n02708093', 'n07716906', 
'n02325366', 'n02025239', 'n02097209', 'n02106662', 'n02277742', 'n02117135', 
'n02087394', 'n01601694', 'n02447366', 'n01682714', 'n03884397', 'n01537544', 'n03063689', 
'n04606251', 'n02493509', 'n02090622', 'n02071294', 'n13037406', 'n04146614', 'n02342885', 
'n02110958', 'n03223299', 'n02963159', 'n02093859', 'n01494475', 'n01955084', 'n02490219', 
'n02840245', 'n02108000', 'n01944390', 'n01860187', 'n02113799', 'n01910747', 'n02086910', 
'n01978455', 'n02107312', 'n02965783', 'n02013706', 'n04033901', 'n01692333', 'n03207941', 
'n02109961', 'n02687172', 'n02002724', 'n01775062', 'n02104365', 'n01749939', 'n01945685', 
'n01704323', 'n04136333', 'n02105855', 'n02443484', 'n02056570', 'n02403003', 'n02134418', 
'n03417042', 'n02096051', 'n02978881', 'n01531178', 'n03065424', 'n01806567', 'n02100877', 
'n03126707', 'n01843065', 'n02814860', 'n02088238', 'n02999410', 'n01484850', 'n02259212', 
'n02097474', 'n02877765', 'n02099712', 'n02123159', 'n01630670', 'n04252077', 'n03218198', 
'n02489166', 'n02727426', 'n02097047', 'n02492035', 'n01728572', 'n03337140', 'n02268853', 
'n01872401', 'n02094433', 'n02206856', 'n01753488', 'n02910353', 'n02114855', 'n03179701', 
'n01498041', 'n04009552', 'n02177972', 'n03016953', 'n02894605', 'n01843383']


def get_imagenet_id_list(class_name):
    id_list = []
    if class_name == 'imagenet-dog':
        for x in pascal_to_synset['dog']:
            synset_id = parse_synset_str(x)
            id_list.append(synset_to_imagenet[synset_id]['imagenet_id'])
    
    elif class_name == 'imagenet-bird':
        for x in pascal_to_synset['bird']:
            synset_id = parse_synset_str(x)
            id_list.append(synset_to_imagenet[synset_id]['imagenet_id'])
    
    elif class_name == 'imagenet-pascal':
        for key in pascal_to_synset:
            for x in pascal_to_synset[key]:
                synset_id = parse_synset_str(x)
                id_list.append(synset_to_imagenet[synset_id]['imagenet_id'])

    elif class_name == 'imagenet-100':
        for synset_id in random_synset_100:
            id_list.append(synset_to_imagenet[synset_id]['imagenet_id'])
    
    else:
        for key in synset_to_imagenet:
            if key != 'background':
                id_list.append(synset_to_imagenet[key]['imagenet_id'])

    return id_list