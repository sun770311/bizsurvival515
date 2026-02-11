## User Stories
1. Entrepreneur
- Wants: To be able to assess if the rent their business is likely to succeed/fail. Maybe the business idea is not the problem, but rather their location or surrounding businesses.
- Interaction methods: Looking up their own business and the surrounding ones. In this way, they can anticipate outcomes and plan ahead.
- Needs: It needs to be updated constantly since we might be talking about a new company. Moreover, it needs to be as accurate as possible
- Skills: Economics, business knowledge. Might be technically inclined as well.
2. Urban Planning Technician
- Wants: An overview of business ecosystems across NYC boroughs, early detection of commercial decline in neighborhoods, discover emerging commercial vitality.
- Interaction methods: Filtering by business category, physical location; examine historical and predicted survival statistics.
- Needs: Interactive spatial map, comparative and filtering dashboard. Accurate geocoding of businesses, open versus closed status over time.
- Skills: Highly technical, is familiar with the technical skills relevant to the backend and interface.
3. Average NYC Citizen
- Wants: See if a certain business (one they like, for example) is projected to survive.
- Interaction methods: Looking up businesses and observing the business forecast
- Needs: The citizen needs to see if a certain business will survive to determine if they’ll frequent and support the establishment, possibly invest, and more.
- Skills: Little to no technical skills and wants to work with simple UI.
4. Property Owner
- Wants: To decide which business they could rent their property to, based on whether they would thrive in that specific area.
- Interaction methods: Providing a list of potential businesses interested in their property and checking which business type ranks the best historically for a particular area.
- Needs: The tool should have business categories/types data and a ranking system based on historical data for different areas.
- Skills: Should know business categories, understand ranking
5. Bank Manager/VC
- Wants: To decide if the business can be lent money based on its likelihood to succeed
- Interaction Methods: Drilldowns from high level portfolio risk to individual business profiles (granularity to be defined), Filter based on multiple conditions, What-if models (Ex: If X happens what impact will it have on Y)
- Needs: The tool should let them view the list 
- Skills: Statistical Analysis, Correlations, Financial Literacy

## Use Cases
Use Case 1: User searches for a specific existing establishment in NYC
- User: Searches for establishment name
- System: User is presented with a search bar and a map of NYC
- User: User starts to type the name of an establishment in the search bar
- System: Auto-generated establishments are recommended as the user types.
- User: User can either click on an auto-generated recommendation or continue to type
- System: Once the user (1) clicks on recommendation or (2) hits “enter” on their search, 
System will provide establishment-specific information to the user.
System will list recommendations, paired with location on NYC map.


Use Case 2: User hovers over pins on spatial map to explore surrounding businesses
- User: Hovers mouse over location pin
- System: Displays pop-up card of business information with license status
- User: User is interested in the displayed business and saves to favorites
- System: Adds business to Starred list and fills in star symbol next to business name


Use Case 3: User searches for a type of business, 
- User: inputs “Barbershop” in the filters
- System: User can see a map with a tab full of filters.
- User: starts typing “Bar”
- System: provides some known options to select. If its a new business type, it shows all the possible options so that the user selects the most similar one.
- User: ticks the type of information they want to see, like geographic coordinates, years since creation, size of establishment.
- System: by default shows all of them but allows the user to untick some of them.
- User: wants to download a picture of the map or download it as a pdf.
- System: shows buttons to download and shows the format.

Use Case 4: User selects the type of business that are currently looking to rent their property from a list and gets an ordered list of the same categories based on the level of success.
- User: Selects relevant business categories
- System: Orders the selected list based on success metrics.
- User: Selects all business categories
- System: Returns a ranked list of the most successful business categories in that particular area.


Use Case 5: Searches for a specific business category (ex: Finance, Automotive)
Search returns a list of business that are tagged with that category
- User: Enters a business category (Dropdowns to help ease the user’s search process)
- System:Returns a list of business in that particular category, also suggests the business categories that are very similar to the category searched
- User: Clicks on a particular business of his interest. 
- System: Shows the financial insights/ numbers of that business, also displays charts showing yearly trends. A map appears showcasing other businesses that are in the vicinity of the chosen business.


