# Response surface optimization.

In analitycal chemestry, response surface methodology (RSM) is common used to 
described analyse the influence of many factor to a response in an analitycal 
method. 

In this case, the important data is allocated in a csv file with the factors 
and responses correctly identified. 

statsred.py read the data and fit a cuadritic linear model for the corresponding 
response. The ANOVA table is presented in console. You can use this data to 
take decision and ommit coefficient it order to enhence the model. The you are
stisfied with the fitted model, the next three plot are allocated in the path
you give. 
    - Coefficient plot
    - Residual plot
    - Prediction vs Experimental plot

optimization.py perform and optimization of the fitted model constrained to the
experimental factors. In this case a CCD design is only valid in the zone 
constrained by the experimental factor (cirumference o sphere). Then, the
response surface is plotted with the eeperimental points and the maximum response.

caca