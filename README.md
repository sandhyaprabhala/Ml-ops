# Ml-ops

Final Exam : </br>
program is in mnist->compare_classifiers.py </br>
Date : 27-11-2021 </br>




![final_exam_1](https://user-images.githubusercontent.com/88537096/143683565-30b842b4-189c-4835-b77c-e38a76212b65.png)

The classifier used is Support Vector Machine. </br>
Here the 2 hyperparameters under consideration are C and Gamma values. </br>
The C parameter indicates how much to avoid misclassifying each training example. </br>
The Gamma parameter is the inverse of the radius of influence of samples selected by the model as support vectors.</br>

For the Train:Dev:Valid split of 70:15:15, C and Gamma are randomly choosen and corresponding Accuracies for Train, dev, and test splits are calculated. </br>

For large values of C, it can be seen that a smaller-margin of hyperplane is choosen, and the corresponding Accuracies are found to be high. This tells us that the hyperplane correctly classifies the training points. <\br>
Gamma decides the curvature for a decision boundary and is set before the model is trained. The range of gamma here is [1e-05,0.0001,0.001,0.01], which is known to give the best results for the given digit dataset. <\br>

When we combine both the hyperparameters, it has been seen that for higher values of C, even the smaller radius of decision boundary (gamma) gives a good accuracy, on th eother hand for smaller values of C, Gamma = 0.01 gives a good accuracy. <\br>



---------------------------------------------------


Assignment - 9 : </br>
program is in mnist->api->hello.py </br>
Date : 18-11-2021 </br>



![Assignment-9(1](https://user-images.githubusercontent.com/88537096/142363169-f63787bc-5f5c-4991-be5c-3512e3141f40.png)

![Assignment-9(2](https://user-images.githubusercontent.com/88537096/142363187-edd5ec2c-dd1f-4eb5-b609-96af2ff9f3b0.png)


Assignment - 8 : </br>
program is in mnist->compare_classifier.py </br>
Date : 8-11-2021 </br>

![A8(1)](https://user-images.githubusercontent.com/88537096/140733692-aa097bc5-9686-4c4a-93f9-2f4b2075180d.jpg) </br>


![A8(2)](https://user-images.githubusercontent.com/88537096/140733727-95e8a253-e715-4588-bd40-5ac628b5d8eb.jpg) </br>


![A8(3)](https://user-images.githubusercontent.com/88537096/140733737-45c08fa5-a6d6-49f9-adff-21246fb2b1e6.jpg) </br>


Assignment - 7 : </br>
program is in mnist->compare_classifier.py </br>
Date : 28-10-2021 </br>

![image](https://user-images.githubusercontent.com/88537096/139225648-eabf1560-1ab0-437f-ba24-09db49dd4a0f.png)


Assignment - 6 : </br> 
Date : 07-10-2021</br>

![image](https://user-images.githubusercontent.com/88537096/136397537-ae2d4f47-9c41-4c9a-a03a-c9073243d37c.png)



