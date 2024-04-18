# CS1440/2440 Final Project: Spectrum Auction

## Introduction

The Spectrum Auction (LSVM18): This final project is your chance to put everything we've learned into practice, especially the cool strategies from the last few labs. It's going to be remote and a lot more free-form than what we're used to, so there's plenty of room to experiment and find what works best for your agent. I hope you all have fun!

## Setup and Installation

Follow these steps to set up your environment and install the necessary package for the lab.

### Step 1: Git Clone the Repository

Open your terminal and navigate to where you want to clone the repository

```bash
git clone https://github.com/brown-agt/Spectrum-Auction-Stencil.git
```

### Step 2: Create a Virtual Environment

Please then navigate to your project directory. Run the following commands to create a Python virtual environment named `.venv`.

If you own a Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If you own a Windows

```bash
python3 -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install the agt server package

```bash
pip install --upgrade pip
pip install --upgrade agt-server
```

## Agent Methods

For the `LSVM Agent`s here are a few methods that you may find helpful!

### Basic Information Methods

- **`get_regional_good()`**:
  - Returns the specific good designated as the regional good for this agent, crucial for regional bidders to focus their strategies around.

- **`get_goods()`**:
  - Retrieves a set of all the goods' names available in the auction, allowing the agent to understand the full scope of what's being auctioned.

- **`is_national_bidder()`**:
  - Checks if the agent is classified as a national bidder, a key distinction that affects bidding strategy and interest in goods.

- **`get_shape()`**:
  - Provides the dimensions of the goods space as a tuple, indicating how goods are organized and potentially their proximity to each other.

- **`get_num_goods()`**:
  - Returns the total number of goods available in the auction, giving the agent a sense of the auction's scale.

- **`get_goods_to_index()`**:
  - Retrieves a dictionary mapping goods names to their indices (tuples), facilitating the conversion between good names and their index locations in valuation arrays.

- **`get_tentative_allocation()`**:
  - Returns a set of strings representing goods tentatively allocated to this agent, showing the current potential gains from the auction.
  
- **`get_current_round()`**:
  - Provides the current round number in the auction, allowing the agent to adjust strategies over time.

- **`get_goods_in_proximity()`**:
  - Lists names of goods within the agent's regional proximity or all goods if a national bidder, guiding regional bidding strategies.

- **`proximity(arr, regional_good)`**:
  - Filters goods based on their proximity to a specified regional good, relevant for regional bidders to evaluate their interest in nearby goods.

#### Utility Calculation Methods

- **`calc_total_valuation(bundle)`**:
  - Calculates the total valuation of a given bundle of goods for the agent

- **`calc_total_prices(bundle)`**:
  - Calculates the total prices of a given bundle of goods for the agent

- **`calc_total_utility(bundle)`**:
  - Calculates the total utility of a given bundle of goods for the agent

#### Valuation and Bidding Methods

- **`get_valuation_as_array()`**:
  - Returns the agent's valuations as a numpy array, offering a structured view of the agent's value assessment of all goods.

- **`get_valuation(good)`**, **`get_valuations(bundle)`**:
  - Retrieves the valuation for specific goods or a set, aiding in determining how much to bid for particular items.

- **`get_min_bids_as_array()`**, **`get_min_bids(bundle)`**:
  - Provides the minimum bids required either as a numpy array (taking in indices) or a map taking in good names. For `get_min_bids` if bundle is not provided it just returns the map for all goods, otherwise a bundle or set of goods can be provided for which you want the min_bids.

- **`is_valid_bid_bundle(my_bids)`**:
  - Checks if a set of bids is valid according to game rules and agent's constraints, preventing invalid bid submissions.

- **`clip_bids(my_bids)`**, **`clip_bid(good, bid)`**:
  - Adjusts bids to meet or exceed the minimum bids, aligning bids with auction rules for either a map of goods to bids, or for a single good and bid.

#### Conversion Methods

- **`map_to_ndarray(map, object)`**, **`ndarray_to_map(arr)`**:
  - Converts between mappings of goods to values and numpy arrays, useful for data manipulation and analysis. This is useful as there are a lot of helper functions that either return `_as_array()` or as a map. This will allow you to easily convert between the two.

### History Methods

Detailed history methods allow the agent to analyze its performance and adjust strategies accordingly. These are returned in a `game_report` that contains a dictionary of the history of the auction that the agent has access to. Both `game_report` and the class has access to the following methods to retrieve different parts of the `game_report`. [E.g. `agent.get_game_report().get_util_history() == agent.get_util_history()`]

- **`get_game_report()`**: Retrieves the Game Report object containing historical information that the agent has access to.

- **`get_util_history()`**: Retrieves a list of the agent's utility values over time, showing how well the agent has been performing.
- **`get_bid_history()`** and **`get_bid_history_map()`**: Offers a history of the bids the agent has made, either as numpy arrays or mappings.
- **`get_price_history()`** and **`get_price_history_map()`**: Gives a record of the prices over the course of the auction.
- **`get_winner_history()`** and **`get_winner_history_map()`**: Details the history of who won what bids in previous rounds, helping the agent to adjust its future bids based on past success rates.

- **`get_previous_util()`**: Provides the most recent utility value for the agent.
- **`get_previous_bid_history()`** and **`get_previous_bid_history_map()`**: Provides the most recent bids the agent has made, either as numpy arrays or mappings.
- **`get_current_prices()`** and **`get_current_prices_map()`**: Provides the most recent prices in the auction.
- **`get_previous_winner_history()`** and **`get_previous_winner_history_map()`**: Provides the most recent history of who won what bids in the last round.

## Notes about the python code

- Please Please Please refer to the final project handout and read through it carefully! It contains a lot of information specific to this implementation of the spectrum auction and making sure your code works for submission.
- In class, I mentioned that you need to do relative imports but that is outdated news, the autograder should automatically handle that for you now so as long as you don't name your files random.py or numpy.py or [Insert Common Package Name].py, the submission should work.
- Please let us know if you want to import any new packages not natively provided in agt_server and we will install it on our end after checking it so that the code can import it in the final submission. [E.g. last time we needed to actually install tensorflow for the tensorflow code to run correctly]
