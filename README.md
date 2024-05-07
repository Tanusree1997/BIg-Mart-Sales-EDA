**Big Mart Sales EDA**

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/a81eadaa-ea71-4cc0-a43c-3a61123598b0)
This project undertakes a comprehensive analysis of Big Mart's sales performance, focusing on item attributes and outlet characteristics. Leveraging Python libraries including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn, exploratory data analysis has been conducted to gain insights into sales trends and patterns. 

**Dataset Link:** https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets

**Tool used:** Google Colab

**Process** 

**Step 1: Data Exploration**

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/d6b1f258-3ffe-4de7-bd3d-2551900c78a7)

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/0e3ab7b3-fa9a-4028-8cef-9e8b366b67b6)

There are 5 numerical and 7 categorical fields.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/fb653259-5245-4236-b143-c4122dbfa2f6)

We can observe an interesting minimum value for Item_visibility because it shows there are some items with zero visibility. This is most likely to be due to glitches in data collection by the stores and should be considered missing values. There are some missing values in Item_weight field also. We can see that Item_MRP and Item_Outlet_Sales do not have any zero or negative values which is good. If we notice the minimum, maximum and quartile values we can understand that the data does not have any outliers.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/d69083ed-924c-4b84-8039-c9be36fb32c2)

There are some missing data in the Outlet_Size field.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/1f6fe275-ad8a-48d9-b00a-d3c2e3817782)

Here, Low fat is written in three different ways and Regular is written in two different ways. Item_fat_Content has actually two unique values Low Fat and Regular.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/11549d29-6cdc-4635-8e4e-9f995552f063)

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/5975c072-df90-4d24-b3e2-74fd69c6ffca)

Descriptive statistics and value_counts() indicates that there are no possibility of outliers in this dataset.

Checking the duplicated rows and missing values in the dataset:

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/963bb6e1-eb23-4aeb-ad3f-f78716d73183)

There is no duplicate row in the data.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/666d9aec-02a0-4137-93ec-8227c4f4f3eb)

The fields with missing values have a very high percentage of missing values and thus we can not remove all the rows with missing values. These two fields are also important parts of the feature list and therefore we can not drop these fields. We have to find the best way for missing value imputation in these fields.

**Step 2: Missing Value Imputation**

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/4a933487-fcf5-4730-aed6-0fcd324cbfda)

Note that, variance of interpolation_weight is closer to the variance of original_data.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/cc618c08-a1ed-4586-b412-2407617d80e4)

It is evident that interpolation method is best suited for missing value imputation in this field.

Let us try the multivariate method for missing value imputation in this field: 

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/6b33d369-a327-4912-93d7-843de0e6cb7a)

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/92c33efb-aed5-4866-8756-a41cc9f49e7a)

Comparing all four methods, we can conclude that Interpolation method is the best for missing value imputation in this data. 

Using the interpolation method to fill the missing values

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/373f13ea-00ae-4c13-a505-2f921e633965)

It is to be noted that, Item_visibility field has some rows with 0.00 value and it's highly improbable for an item's visibility to be precisely zero. Visibility typically refers to the prominence or exposure of an item on store shelves or within a retail environment. Even if an item is placed in a less visible location, it would still have some level of visibility. However, it's conceivable that an item's visibility could be extremely low, approaching zero, if it's stored in a completely obscured or inaccessible location within the store. Nevertheless, such occurrences are rare and could indicate operational issues or errors in data collection.

Therefore, we need to fix this. For simplicity, the zero values are replaced by nan values.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/8bab14d9-4e4f-402a-9926-a3741539677a)

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/50835a15-218c-4fde-9bf3-c0f00de0e821)

Comparing all three methods, we can conclude that Interpolation method is the best for missing value imputation in this field. Using the interpolation method to fill the missing values:

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/1d26b153-d40a-4520-9b91-fee7c3a44db0)

To fill the gaps in the Outlet_Size field it is important to note that Outlet_Size is dependent on Outlet_Type (for example, grocery stores are supposed to be smaller than the supermarkets). Therefore, we will first look at the distribution of Outlet_Size by Outlet_Type using a Pivot table.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/c34a2554-8cd6-41fe-8b8a-265c128ee0cb)

It should be noted that most of the Grocery store and Supermarket Type 1 are of small sizes. 

Now we will look at the distribution of the missing Outlet_Size values based on Outlet_Type:

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/2e2233a5-382e-4a14-8e23-5b9af6ad74b8)

The missing values in the Outlet_Size field is associated with Grocery Store and Supermarket Type1. Therefore, we have filled the missing values in this field with small outlet_size.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/957ad0bd-5b64-410a-9109-5c658a959267)

**Step 3: Data Preprocessing** 

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/74ee4300-13ac-4e7e-a3e1-a844e0510376)

Converting the Outlet_Establishment_year to Outlet_age

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/aa722453-37fa-4de4-86e3-e48846c0e913)

**Step 4: Data Visualization**

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/9b07166b-b7a3-4c8b-9ed1-9b8cc31d5b8c)

Item Weight and Item Visibility has almost no effect on Item_Outlet_Sales. 

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/387839a1-0173-4bd4-bf10-b660ba3dc03a)

Low_fat products are more popular among people as this type of products generate more sales.

Similarly we have,
![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/eef3c2d9-7938-4b1f-8334-8e84bb0e85f8)

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/2fb7f55b-25ca-46c1-b97c-44ef8eb4e304)

More sales is generated by the Outlets with “Small” Outlet Size which means people spend more on Grocery stores and Supermarket Type1.

Outlets in Tier1 locations generated more sales.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/32207f55-a2a6-4b89-b31b-dab7ea36db02)

Supermarket Type1 and Grocery Stores generate a higher sales than Supermarket Type 2 and 3.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/d70b6cee-7c89-4634-83e5-27e1eeb59e7a)

Sales is not dependent on the age of the Outlet. This means, popularity of the outlets are not determined by its age.

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/b1e4c893-d4f9-47db-84e4-1b3a5e02956e)

The top 5 item categories based on sales are Fruits and Vegetables, Snack Foods, Household, Frozen food, and Dairy (and Canned foods).

![image](https://github.com/Tanusree1997/BIg-Mart-Sales-EDA/assets/164666871/4333260f-a5fc-4c4c-aa60-b6039cedd5e3)

The top 5 Outlets based on sales are identified by the outlet Identifiers: OUT027, OUT013, OUT049, OUT046 and OUT035. Two outlets are performing really bad which are- OUT019 and OUT010. 

**Conclusion**

This project concludes by identifying the patterns of sales based on the attributes of the products and the outlets of Big Mart based on exploratory data analysis and data visualizations. 




