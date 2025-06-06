# define supporting functions

import torch 
import torchvision.models as models
from torch import nn
import open_clip
from torchvision import transforms

# print configurations
def print_args(opt):
    
    message = ''
    message += '='*46 +' Options ' + '='*46 +'\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''

        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '='*48 +' End ' + '='*47 +'\n'
    print(message)

# load model on device, get number of classes
def loading_models(args):
    if args.model_name == "ResNet":
        net = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        net.fc = nn.Linear(net.fc.in_features, args.num_classes)
        net = net.to(args.device)
        model = torch.load(args.model_filepath_1, map_location=args.device) #original
        net.load_state_dict(model)

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if args.model_name == "CLIP":
        import sys
        sys.path.append('../')
        sys.path.append('../BadMerging/')
        sys.path.append('../BadMerging/src')

        from BadMerging.src.modeling import ImageClassifier


        model = torch.load(args.model_filepath_1 ,weights_only=False).to(args.device)
        encoder, train_preprocess, val_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
        model.model.transformer = encoder.transformer.to(args.device)

        classification_head = torch.load(args.model_filepath_2, weights_only=False).to(args.device)
        net = ImageClassifier(model, classification_head)

        transform = train_preprocess

    return net, transform




def classes_matching(target_classes_all,triggered_classes_all):
    
    start_index = len(target_classes_all)
    for i in range(len(triggered_classes_all)):
        tmp = triggered_classes_all[i]
        for sss in range(tmp.size(0)):
            target_classes_all.append(target_classes_all[i])
            triggered_classes_all.append(tmp[sss])



    end_index = len(target_classes_all)



    if start_index != end_index:
        target_classes_all = target_classes_all[start_index:]
        triggered_classes_all = triggered_classes_all[start_index:]


    return target_classes_all, triggered_classes_all


def identify_trigger_type(raw_target_classes,raw_victim_classes):

    if raw_victim_classes != None:
        target_classes, victim_classes = classes_matching(raw_target_classes, raw_victim_classes)
        num_classes = len(victim_classes)
        trigger_type = 'polygon_specific'

        print(f'Trigger Type: {trigger_type}')
        Candidates = []
        for i in range(len(target_classes)):
            Candidates.append('{}-{}'.format(target_classes[i],victim_classes[i]))
        print(f'Target-Victim Pair Candidates: {Candidates}')

    
    else:
        #print(raw_target_classes)
        if raw_target_classes != None:
            num_classes = len(raw_target_classes)
            target_classes = raw_target_classes
            victim_classes = raw_victim_classes
            trigger_type = 'polygon_global'
            print(f'Trigger Type: {trigger_type}')
            print(f'Target class: {target_classes} Victim Classes: ALL')
        
        else:
            target_classes = raw_target_classes
            victim_classes = raw_victim_classes
            num_classes = 0
            trigger_type = 'benign'


    return target_classes,victim_classes,num_classes,trigger_type
    

def trojan_det(args,trigger_type,l1_norm,sym_l1_norm):


    if trigger_type == 'polygon_global':
        
        if l1_norm < args.global_det_bound:
            return 'trojan'

        else:
            return 'benign'

    elif trigger_type == 'polygon_specific':
        

        if l1_norm < args.local_det_bound:

            if args.sym_check and sym_l1_norm / l1_norm > args.ratio_det_bound:
                return 'trojan'
            

            else:

                return 'benign'
        
        else:
            return 'benign'




