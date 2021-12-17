# Potato Leaf Disease Classification

##### Profile Visits : 
![Visitors](https://visitor-badge.glitch.me/badge?page_id=ronylpatil.Potato-Leaf-Disease-Classification&left_color=lightgrey&right_color=brightgreen&left_text=visitors) 

<p align="center">
  <img class="center" src ="https://thepracticalplanter.com/wp-content/uploads/2021/09/IS-Potato-Plant.jpg" alt="Drawing" style="width: 1400px; height: 600px">
</p>

<b>Description : </b> Here I used __Artificial Intelligence__ in diagnosing plant diseases. Various diseases like early blight and late blight immensely influence the quality and quantity of the potatoes and manual interpretation of these leaf diseases is quite time-taking and cumbersome. Therefore I created a __Web App__ using <b>Streamlit</b> which simply classify <b>Potato Leaf Diseases</b> and, finally deployed the Web-app on __Heroku__. Internally, our model is built using a simple <b>Convolutional Neural Network Architecture</b> to classify <b>Potato Leaf Diseases</b>. Initially I collected ready-made data from internet. Then due to small size of dataset, I used one of the simple and effective method, called <b>Data Augmentation</b> to increase the size of dataset as well as to reduce overfitting of our model. At the end built a __Deep Learning Model__ to detect or classify Potato Leaf Diseases and got a __test accuracy of 97%.__

<b>Heroku App : https://potato-leaf-disease-detection.herokuapp.com/</b><br>
<b>Dataset Source : https://www.kaggle.com/arjuntejaswi/plant-village</b><br>
<b>Article Link : https://www.analyticsvidhya.com/blog/2021/12/end-to-end-potato-leaf-disease-prediction-project-a-complete-guide/</b>

<b>Folder Structure : </b>
```
                    Potato Leaf Dataset       --> main folder
                      ----| train      
                          ----| Potato_Healthy
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Early_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Late_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg

                      ----| test      
                          ----| Potato_Healthy
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Early_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Late_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                              
                      ----| valid      
                          ----| Potato_Healthy
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Early_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
                          ----| Potato_Late_Blight
                              ----| img1.jpg
                              ----| img2.jpg
                              ----| img3.jpg
```

<b>Sample Output : </b> The output is showing 3 thing's. 
* <b>Predicted Class : </b>The model's output.
* <b>Actual Class : </b>The actual output.
* <b>Confidence : </b>How confident our model is.
 
https://user-images.githubusercontent.com/63307564/145527457-cd9f8844-fe5d-47d2-ba18-759bdc667489.mp4
  
<p align="center">
  <img class="center" src ="/main/sample/potato.png" alt="Drawing" style="width: 1400px; height: 800px">
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/ronylpatil/">Made with ❤ by ronil</a>
</p> 

<!-- © 2021 Ronil Patil<br>
[![Website](https://img.shields.io/badge/Made%20with-%E2%9D%A4-important?style=for-the-badge&url=https://www.linkedin.com/in/ronylpatil/)](https://www.linkedin.com/in/ronylpatil/) -->
