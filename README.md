# Pro_CCaps
Automatic image colourisation studies how to colourisegreyscale  images.    Existing  approaches  exploit  convolu-tional layers that extract image-level features learning thecolourisation  on  the  entire  image,  but  miss  entities-levelones due to pooling strategies.  We believe that entity-levelfeatures are of paramount importance to deal with the in-trinsic multimodality of the problem (i.e., the same objectcan have different colours, and the same colour can havedifferent properties).  Models based on capsule layers aimto identify entity-level features in the image from differentpoints of view, but they do not keep track of global features.Our network architecture integrates entity-level featuresinto  the  image-level  features  to  generate  a  plausible  im-age colourisation.  We observed that results obtained withdirect  integration  of  such  two  representations  are  largelydominated  by  the  image-level  features,  thus  resulting  inunsaturated  colours  for  the  entities.   To  limit  such  an  is-sue,  we  propose  a  gradual  growth  of  the  reconstructionphase  of  the  model  while  training.By  advantaging  ofprior knowledge from each growing step, we obtain a sta-ble collaboration between image-level and entity-level fea-tures that ultimately generates stable and vibrant colouri-sations. Experimental results on three benchmark datasets,and a user study, demonstrate that our approach has com-petitive performance with respect to the state-of-the-art andprovides  more  consistent  colourisation.

This Project is going to appear at WACV2022.

Please if you use this repository for you research, cite me:\
"Pro-CCaps: Progressively Teaching Colourisation to Capsules"\
Rita Pucci, Christian Micheloni, Gian Luca Foresti, Niki Martinel\
WACV 2022
