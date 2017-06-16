
# walk.py

Computes a random vector and then generates many sentences with a small noise
perturbation. It prints new sentences and highlights how they differ from the original. 


## Examples


__A man goes in front of a skateboard .__

A __snowboarder crashes__ in a __board at the hospital__ . 

A man goes in a __cabin__ . 

A man __pours into the mud__ . 

A man goes in __midair__ .

A man goes __into the mud__ . 

A man goes in front of a __telescope__ . 

A man goes in __their driveway__ . 

A man goes in . 

A man goes in front of __the__ skateboard . 

A __snowboarder crashes__ in front of a __bed__ . 

A man goes in __the mud__ . 

A man __pours__ in front of a __lake__ . 

```
Workers clean a very colorful street .
Workers *enjoy* a very colorful . 
*Women enjoy* a very *quiet* . 
Workers clean a *long white area* . 
Workers *enjoy* a very colorful *hallway* . 
Workers *carry* a very colorful *and empty* . 
Workers clean a *long white structure* . 
Workers *enjoy* a very colorful *narrow* . 
Workers clean a very *tall structure* . 
*Women enjoy* a *green sunny* . 
Workers clean a *long white balloon* . 
*Women enjoy* a *green outdoor* . 
*Women enjoy* a *green sunny weather* . 
Workers *enjoy* a very colorful *environment* . 
Workers *enjoy* a *green construction and* . 
*Women enjoy* a *green outdoor staircase* . 
Workers *enjoy* a very colorful *walkway* . 
*Women enjoy* a very *green hallway* . 
Workers *enjoy* a very colorful *city* . 
```

# vector.py

An experiment to try to change the content of a sentence based my modifying the
latent variable z. The alteration is done by sampling sentences from a fixed
z, then updating towards the mean of the desired feature. We do this repeatedly
until the argmax generation has the correct feature.


## Requirements

* spacy

## Usage

1) Generate 1M sample sentences with vectors and features

> python -m experiments.vector gen --dump  features.pkl --load_path maxlen15

2) Attempt to alter new sentences based on the mean vector of `VERB_standingssss` sentences.

> python -m experiments.vector alter --dump  features.pkl --alter VERB_standing --load_path maxlen15


## Examples



```
Sent  0 : 	  The man is wearing a scarf 	
Sent  1 : 	  The man is wearing a hat 	
Sent  2 : 	  The man is smoking 	
Sent  3 : 	  man is seated 	
Sent  4 : 	  man is standing on roof . 	


Sent  0 : 	  A man rode on vacation . 	
Sent  1 : 	  A man is riding . 	
Sent  2 : 	  A man is stand on . 	
Sent  3 : 	  A man is standing on fire . 	


Sent  0 : 	  A brown dog is inside of the house . 	
Sent  1 : 	  A white bird is sitting at <oov> . 	
Sent  2 : 	  A white boy is working in water . 	
Sent  3 : 	  A white man is sitting and is in a pool . 	
Sent  4 : 	  A young dog is standing in a water . 	


Sent  0 : 	  A man in a cowboy hat with sand near a mirror . 	
Sent  1 : 	  A man in a hand hat with face sits in a piece . 	
Sent  2 : 	  A man in a hat standing in hand as a piece of works . 	


Sent  0 : 	  A girl has a beach on blanket . 	
Sent  1 : 	  There is a boys sleeping on . 	
Sent  2 : 	  There is a standing girl on the shoreline . 	


Sent  0 : 	  A people are sitting on a close trail . 	
Sent  1 : 	  Boys are sitting with a tank top . 	
Sent  2 : 	  people are standing up a big . 	


Sent  0 : 	  There is the road . 	
Sent  1 : 	  There is a <oov> . 	
Sent  2 : 	  There is a man . 	
Sent  3 : 	  There is a man in a street . 	
Sent  4 : 	  There is a man in a street . 	
Sent  5 : 	  There is a man in a street . 	
Sent  6 : 	  There is a man standing in a street . 	


Sent  0 : 	  A young girls jump fast and a wheelchair of an outside . 	
Sent  1 : 	  A football teams stand next to a brown building and a street . 	
Sent  2 : 	  Several girls are standing up and a beautiful of a street . 	


Sent  0 : 	  Two men working on an outdoor bus beside a orange house and orange . 	
Sent  1 : 	  Two men sitting and conversing through a tall grass playing together . 	
Sent  2 : 	  Two men standing outside near a dead and man standing around wearing vests . 	


Sent  0 : 	  A rock plastic glass and biting the branch of it . 	
Sent  1 : 	  A very young boy hanging out of the camera and tall . 	
Sent  2 : 	  A very sad boy standing out of the tall grass . 	


Sent  0 : 	  A lady playing hopscotch on her hands . 	
Sent  1 : 	  A girl playing hopscotch on the sidewalk holding . 	
Sent  2 : 	  A women is sitting behind the window . 	
Sent  3 : 	  A woman standing inside of the grass . 	


Sent  0 : 	  The woman is smiling . 	
Sent  1 : 	  The woman is smiling . 	
Sent  2 : 	  The woman is standing . 	


Sent  0 : 	  Boy in orange on a sand beach . 	
Sent  1 : 	  Boy on holding a sand beach in pool . 	
Sent  2 : 	  Boy on holding a snow ball in is standing . 	


Sent  0 : 	  A man rides the motorcycle or fire coming across the water . 	
Sent  1 : 	  The man is in winter gear as the traffic lights water from the crowd . 	
Sent  2 : 	  The man is behind the river or trying to make water into the distance . 	
Sent  3 : 	  The man is standing behind through the side of or preparing the water . 	


Sent  0 : 	  Two dogs fight 	
Sent  1 : 	  Two men walk out 	
Sent  2 : 	  Two men are out 	
Sent  3 : 	  Two men are walking around a pool . 	
Sent  4 : 	  Two men are walking a fountain . 	
Sent  5 : 	  Several men are walking by a fountain . 	
Sent  6 : 	  Several men are standing by a fountain . 	


Sent  0 : 	  Several dogs play a . 	
Sent  1 : 	  There are two sitting . 	
Sent  2 : 	  There are two sitting in a working . 	
Sent  3 : 	  There are men standing near a . 	


Sent  0 : 	  A boy without his mouth climbs . 	
Sent  1 : 	  A boy wears sunglasses with some white sunglasses . 	
Sent  2 : 	  A boy wall is without his orange . 	
Sent  3 : 	  A young boy is standing under an orange . 	


Sent  0 : 	  five people 's sliding down a boat . 	
Sent  1 : 	  five people kicking the ball at a soccer game . 	
Sent  2 : 	  five people kicking the ball at a soccer game . 	
Sent  3 : 	  five people looking at the ball through a play . 	
Sent  4 : 	  five people dressed as the child play . 	
Sent  5 : 	  five people standing next to the play . 	


Sent  0 : 	  The <oov> are blue . 	
Sent  1 : 	  The white bike are at the court . 	
Sent  2 : 	  The white are standing on the sidewalk . 	


Sent  0 : 	  Men are sitting at a table . 	
Sent  1 : 	  Men are sitting on . 	
Sent  2 : 	  Men are standing on a bench . 	


Sent  0 : 	  A group of men passing them are going a . 	
Sent  1 : 	  A men of men are walking along a street . 	
Sent  2 : 	  A man dressed as two are taking pictures to a run . 	
Sent  3 : 	  A man dressed as three men are over to a river . 	
Sent  4 : 	  Several men standing next to two bikes are near a line . 	


Sent  0 : 	  A crowd is breaking up on some kind of playing . 	
Sent  1 : 	  A crowd is standing around on the balcony and making on . 	


Sent  0 : 	  a person hangs on a wall 	
Sent  1 : 	  A person hangs on a wall on cliff . 	
Sent  2 : 	  A person sits on a wall on it . 	
Sent  3 : 	  A person is standing on a rope on roof . 	


Sent  0 : 	  The kid a bucket . 	
Sent  1 : 	  The kid is nearby . 	
Sent  2 : 	  The kid is nearby . 	
Sent  3 : 	  The kid is standing . 	


Sent  0 : 	  A woman is a man in a beach next . 	
Sent  1 : 	  A man is talking in a field with a glass . 	
Sent  2 : 	  A man is standing in a beach with a garbage . 	


Sent  0 : 	  The young girl sit along on two wooden , enjoying a wrestling . 	
Sent  1 : 	  The young girl sit along side on a wet beach bowling . 	
Sent  2 : 	  Women standing , are sitting on two hanging around a young woman . 	


Sent  0 : 	  There are men sitting on a ledge . 	
Sent  1 : 	  There are men sitting on a ledge . 	
Sent  2 : 	  There are man standing on a ledge . 	


Sent  0 : 	  a blond girl with a painting camera outside 	
Sent  1 : 	  a women with a purse smoking and reading through her . 	
Sent  2 : 	  A woman standing with a smoking and a vehicle holding road . 	


Sent  0 : 	  There are a sad woman on a floor . 	
Sent  1 : 	  There are a sad woman on stairs . 	
Sent  2 : 	  There are standing on a phone with a standing . 	


Sent  0 : 	  There are two cars at a track . 	
Sent  1 : 	  There are two people down a river . 	
Sent  2 : 	  There is two people walking for a sport . 	
Sent  3 : 	  There is two people walking for a big . 	
Sent  4 : 	  There is two people walking on a stream . 	
Sent  5 : 	  people is walking on a stream . 	
Sent  6 : 	  people is walking on a stream . 	
Sent  7 : 	  People are standing by a parade . 	


Sent  0 : 	  Young women are sitting and crossing a street performer . 	
Sent  1 : 	  The colorful ladies are walking toward a couple street . 	
Sent  2 : 	  The naked standing boy are sitting at a street construction . 	


Sent  0 : 	  There is some men walking <oov> walking and looks . 	
Sent  1 : 	  There is a woman standing <oov> walking and looking out . 	
```
