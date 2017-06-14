

## Requirements

* spacy

## Usage

1) Generate 1M sample sentences with vectors and features

> python -m experiments.vector gen --dump  features.pkl --load_path maxlen15

2) Attempt to alter new sentences based on the mean vector of VERB_driving sentences.

> python -m experiments.vector alter --dump  features.pkl --alter VERB_standing --load_path maxlen15


## Examples

"""
Sent  0 : 	  A people are sitting on a close trail . 	
Sent  1 : 	  Boys are sitting up a volleyball . 	
Sent  2 : 	  people are standing up a big . 	


Sent  0 : 	  There is the road . 	
Sent  1 : 	  There is a <oov> . 	
Sent  2 : 	  There is a man in a street . 	
Sent  3 : 	  There is a man in a street . 	
Sent  4 : 	  There is a man standing in a street . 	


Sent  0 : 	  A young girls jump fast and a wheelchair of an outside . 	
Sent  1 : 	  A men stand out next to a green street some . 	
Sent  2 : 	  Several girls standing next to a brick building and a white . 	


Sent  0 : 	  Two men working on an outdoor bus beside a orange house and orange . 	
Sent  1 : 	  Two men sitting and conversing through a tall wall outside of sports . 	
Sent  2 : 	  Two men standing outside , one man with a colorful umbrella are outside gathering . 	


Sent  0 : 	  A rock plastic glass and biting the branch of it . 	
Sent  1 : 	  A very young boy hanging out of the camera and tall . 	
Sent  2 : 	  A very standing boy leaning on the edge of all it . 	


Sent  0 : 	  A lady playing hopscotch on her hands . 	
Sent  1 : 	  A girl playing hopscotch with the ground . 	
Sent  2 : 	  A women is sitting behind the window . 	
Sent  3 : 	  A woman standing in front of it is getting . 	


Sent  0 : 	  The woman is smiling . 	
Sent  1 : 	  The woman is standing . 	


Sent  0 : 	  Boy in orange on a sand beach . 	
Sent  1 : 	  Boy on holding a sand beach in pool . 	
Sent  2 : 	  Boy on holding a boy is standing in front of . 	


Sent  0 : 	  A man rides the motorcycle or fire coming across the water . 	
Sent  1 : 	  The man is in winter gear as the traffic lights water from the crowd . 	
Sent  2 : 	  The man is holding up the landscape object near a body of water . 	
Sent  3 : 	  There is standing up of the side near boats through a block of water . 
"""