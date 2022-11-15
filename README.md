# Intent-based-Chatbot
An automated open-source chatbot that classifies the intents of user input using a custom built neural network classifier, where a set of predefined intents is used to generate the right response after classification. 
<br><br>
The <strong><i>intent.json</i></strong> file contains the required tags, patterns, and responses for classification.

<hr style="width:25%;">
<h2>Demo</h2>
<img src="https://github.com/eshaan2411/Intent-based-Chatbot/blob/main/samples/chatbot_demo.gif">

<hr style="width:10%;">

<img src="https://github.com/eshaan2411/Intent-based-Chatbot/blob/main/samples/demo.png">

<hr style="width:25%;">
<h3>Run in a local system</h3>
<ul>
  <li><strong>Step 1. </strong>Clone the repository</li>
  <li><strong>Step 2. </strong>Run the <i>chatbot_application.py</i> file</li>
</ul>

<h3>Train using Custom intents</h3>
<ul>
  <li><strong>Step 1. </strong>Clone the repository</li>
  <li><strong>Step 2. </strong>Add the required tags, patterns, and responses in <i>intent.json</i> file</li>
  <li><strong>Step 3. </strong>Delete the <i>words.pkl</i>, <i>classes.pkl</i>, and <i>chatbot.h5</i> files</li>
  <li><strong>Optional - </strong>You can change or play with Neural Network Architecture to suit your requirements</li>
  <li><strong>Step 4. </strong>Run the <i>model.py</i> file to train your model on custom intents</li>
  <li><strong>Step 5. </strong>Run the <i>chatbot_application.py</i> file</li>
</ul>
