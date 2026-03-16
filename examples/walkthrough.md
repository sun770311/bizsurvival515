# How to Use the Streamlit Application

A step-by-step walkthrough of the NYC business survival interactive web application, designed to support entrepreneurs, NYC residents, economic researchers, and urban analysts in exploring business risk and neighborhood complaint patterns.

# Table of Contents

1. [Landing Page](#landing-page)
2. [Exploring the Map](#exploring-the-map)
3. [View Model Insights](#view-model-insights)
4. [Create and Simulate](#create-and-simulate)
5. [View Findings](#view-findings)

## Landing Page

Navigate to the application website and click 'Explore Now' to go to the map.

![drone shot](screenshots/landing-page.png)

## Exploring the Map

* Primary user: curious NYC citizen
* Secondary users: entrepreneurs, economic researchers, urban analysts

We see the business landscape explorer powered by Mapbox. Each point on the map is a unique business ID in New York City. A unique ID can correspond to one or multiple business licenses. On the left is a dashboard where users can adjust filters to change the businesses shown. 

![map intro](screenshots/map-intro.png)

For example, let's filter to businesses in Manhattan or Brooklyn.

![filter no card](screenshots/filter-no-card.png)

Clicking on 'Reset filters' will deselect all filters.

To inspect a particular business ID, click on a point on the map and a card will pop up with two sections. The first includes activity status, license expiration date, total number of complaints in history, and license count. The second is a list of one or multiple license records with the business name, address, borough, license number, and contact information.

![map card](screenshots/map-card.png)

Now let's combine filtering and ID selection. Here is an example of all business IDs in Queens which are currently active, holds multiple licenses, and has at least 21 total complaints. One example is JK Tech and Sport Corp. which is both an electronics store and electronic & home appliance service dealer.

![filter card](screenshots/filter-card.png)

## View Model Insights

* Primary users: economic researchers, urban analysts
* Secondary users: entrepreneurs, curious citizens

Click on the sidebar to navigate through the site. The 'logistic regression' page has two tabs: survival simulator and model insights below an overview of what the model predicts, how it was trained, and what the baseline means. The logistic regression model estimates the probability that an NYC business will survive for at least 3 years using aggregated features from its first year.

Let's click on 'Model Insights' first. 

![log intro](screenshots/log-intro.png)

We see the top factors influencing survival probability. We've color coded the predictor types. The top positive and negative coefficients include a mix of predictor types. Take note of the illegal parking and electronics store predictors. These will be used when simulating a hypothetical business later.

![log outputs](screenshots/log-outputs.png)

Below the coefficients table, we see a summary of evaluation results containing class-specific performance, dataset sizes with resampling, and mean predicted probabilities by true outcome.

![log eval](screenshots/log-eval.png)

Now let's navigate to the two Cox survival models. The standard model uses business characteristics from its first month only to visualize survival probability trajectory in the next 1, 3, 5, and 10 years. The time-varying cox model estimates a business's current relative closure risk at each given time point. We'll visualize that more clearly later.

![cox intro](screenshots/cox-intro.png)

Scrolling to the bottom of each tab, we see the Cox model insight. For the standard model, influential predictors are overwhelmingly business category-related, while those for the time-varying model are dominated by complaint-related ones on one side, and business category-related on the other.

![standard cox outputs](screenshots/standard-cox-outputs.png)

![time cox outputs](screenshots/time-cox-outputs.png)

## Create and Simulate

* Primary user: entrepreneurs
* Secondary users: economic researchers, urban analysts, citizens

The logistic regression simulator supports the creation of one hypothetical business. Begin typing in the fields and the system will dynamically filter to match results. 

![log typing](screenshots/log-typing.png)

Here is the completed business profile: an electronics store with one active license and 3 counts of illegal parking complaints in the first 12 months. The user-input latitude and longitude must be within the geographic constraints of New York City and will be assigned to a location cluster. Hover over the hint (question mark icon) for acceptable coordinate values. Click on 'Predict 36-month survival probability'.

![log example ready](screenshots/log-example-ready.png)

Yay! Our business is predicted to survive for at least 3 years. The output compares to the baseline and reiterates the entered and derived feature values used. You'll remember that illegal parking is a predictor associated with increasing the probability of survival, while the business being an electronics store decreases the probability of survival. The sufficient counts of illegal parking complaints is a stronger factor here.

![log example results](screenshots/log-example-results.png)

Our second supported simulator uses the standard cox model to compare one or multiple hypothetical businesses. 

![standard cox multi](screenshots/standard-cox-multi.png)

Let's compare a third party food delivery service with 2 counts of dirty condition complaints against a business that holds licenses for both tow truck company and garage and parking lot. 

![standard cox example ready](screenshots/standard-cox-example-ready.png)

The third party food delivery service is expected to perform much worse compared to the baseline, while the second business performs better. Scrolling down to the coefficient table, our results match the top features increasing and decreasing hazard.

![standard cox graph](screenshots/standard-cox-graph.png)

![standard cox risk summary](screenshots/standard-cox-risk-summary.png)

Our final supported simulator uses the time-varying cox model to randomly generate and compare timelines of multiple hypothetical businesses. Random generation avoids the user having to tediously enter business information at multiple time points.

![time cox multi](screenshots/time-cox-multi.png)

Clicking on the left button generates business states.

![time cox first seed](screenshots/time-cox-first-seed.png)

Clicking on the right button regenerates combinations with the same number of businesses and number of time points.

![time cox new seed](screenshots/time-cox-new-seed.png)

Scrolling down, we see the graph output of relative closure risk across time points as well as a corresponding table of numbers. This illustrates the instantaneous risk that the business will close at the given time.

![time cox graph](screenshots/time-cox-graph.png)

## View Findings

We hope you enjoyed exploring our application! For more detailed conclusions and dataset information, please see the 'findings' page.

![findings overview](screenshots/findings-overview.png)
![findings data](screenshots/findings-data.png)





