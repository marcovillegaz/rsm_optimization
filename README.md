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

## screening.py (remake the figures)
This script perform analytical analisis for the screening step en DLLME for 
factor extractant solvent (HDES) and centrifugue time. The code perfom an 
Analysis of Varience and Tuke HSD test to indetify significative difference between groups. The a box plot is created for each factor to visualize the effect in the repsonse Enrichment factor

## response_surface_fit.py (revisit plotting functions)

## cosmo_screening.py 
This script create a bar plot with the result obtaining from the COSMO-RS
screening between PCB77 and different HDES and save it in the same folder.

## univar_analysis.py
This scripts perform the univariate anaylsis for the four extractant candidates and the centrifuge time.
ANOVA analysis and ad hoc test are realized. Then the result are plotted into a box plot. 

## EF_surface.py
Create a surface plot based on the results obtained. This scripts is only for maintaoinign constant one effect. Due the multidimension array
when computing the predicted response.
## REC_surface.py