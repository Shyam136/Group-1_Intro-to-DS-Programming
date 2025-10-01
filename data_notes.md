# This notebook contains all processes to import/cleanse/validate and prepare data for processing for the remainder of the project.

## Bring in raw dataset

### check shape for validation throughout process 
(7668, 25)

## Preliminary investitation 

### Dataset conventions
* Are gross and budget amounts in US dollars or foreign currency for foreign films?  
     * Confirmed that all amounts shown are in US dollars by comparing documented gross within the individdual IMDB pages for films, which shows US $$, and confirmed that total matches the dataset
* compare the budget/gross to historic online data to determine if the inflation adjustments were already done
    * "Jaws 3D" shows $88M gross in both dataset and on wikipedia https://en.wikipedia.org/wiki/Jaws_3-D
    * "Things are Tough All Over" shows $21M gross in both dataset and wikipedia https://en.wikipedia.org/wiki/Things_Are_Tough_All_Over
    * This seems to indicate that the dataset has not already been adjusted for inflation, meaning we need to as part of our analysis

* Determine if data was inconsistently entered in a way that might separate attribution; e.g. Star Robert Redford (8), Star Robbert Redford (1); 
    * do this for fields ['star', 'writer', 'director', 'country', 'genre', 'company']


* Determine if there are duplicated movie titles
    * any duplication of titles is due to remakes/sequels
 
 ## Data Shaping

### Create separate field for release year in order to join to the inflation multiplier data
* clean the release date to be a standard date by separating into
    * release_date
    * country

### Add field for [decade] in order to reduce the dimensionality of year

### Adjust gross budget and revenue dollars for inflation
* Merged data with ./data/raw/adjusted_dollars.csv to create new fields [adjusted_gross] and [adjusted_budget]

### Create discrete fields for budget and gross rounded to nearest $1M to reduce dimensionality

## Data validity checks

### In order to attribute box office $$ to any particular star, we have to ensure that any one star is recorded exactly the same, e.g. "Robert DeNiro" is not also listed as "Robert De Niro". To do this we will get a list of unique star names and do a vector angle analysis to give similarity score with threshhold >=.75. We need to do the same for writer, director, country, genre and company as well

* the function similarity_check allows us to run this check on individual fields 

## Data validity checks

### In order to attribute box office $$ to any particular star, we have to ensure that any one star is recorded exactly the same, e.g. "Robert DeNiro" is not also listed as "Robert De Niro". To do this we will get a list of unique star names and do a vector angle analysis to give similarity score with threshhold >=.75. We need to do the same for writer, director, country, genre and company as well

* the function similarity_check allows us to run this check on individual fields 

### Analysis of the companies reveals that major studios have subsidiaries that should be folded in with their parent companies and/or have evolving names over time which should be historically joined for the purpose of analysis.

* For each of the company updates, I researched if the companies were in fact related and only update if there is a definitive link.

## Scope Considerations

### Determine how many movies are foreign vs. domestic

#### there are 6735 domestic movies and 933 foreign movies, meaning we have a critical mass of data even without the foreign movies.  The decision was made to remove them.

### Movies with null budget or gross


#### only 132 movies have null gross values, and we cannot consider them for analysis
#### 1607 movies have null budgets, but we will leave them in case budget does not have a significant impact on gross

## write cleaned dataset to ./data/process/movies_model.csv

# Summary of changes between source data and model

## Removed Data
* Removed records with null gross values
* Removed records of movies not produced in United States
    * field [is_domestic] added as boolean to faciliate removal
## Removed Fields
* None; All original fields exist with original values for the purpose of validation
## Additional Fields
* Original Field [released] contained both the release date and country.  That data has been separated into discrete fields
    * [release_date] - date-formatted release data
    * [country] - country of origin
* For the purposes of reduced dimensionality along year, we added [decade] to group 1980's, 1970's, etc.
* To allow for analysis over a century, dollar values had to be adjusted for inflation.  To do this, we added fields:
    * [adjusted_gross] - which is the inflation-adjusted gross value
    * [adjusted_budget] - which is the inflation-adjusted budget value
* To faciliate cleaner regression modeling, we added a field to round budget and gross $$ to the nearest million:
    * [gross_discrete] - gross $$ rounded to nearest million
    * [budget_discrete] - budget $$ rounded to nearest million
* Movie studios ([company]) values which are related by virtue of evolution or parent/subsidiare relationship need to be grouped for proper attribution:
* [grouped_company] - grouped name for varius related studios, e.g.[company] values "Disney Studios" & "Disney Animation" have [grouped_company] value "Disney"


