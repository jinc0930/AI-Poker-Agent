import math

def card_to_num(card_str):
    """
    Convert a card string to a tuple of (rank, suit)
    :param card_str: Card string in format like 'H10', 'CA', 'D7', etc.
    :return: Tuple with (rank_index, suit_index)
    """
    suit_str = "CDHS"
    rank_str = "23456789TJQKA"

    # Handle two-character and three-character card strings
    if len(card_str) == 2:
        suit = suit_str.index(card_str[0])
        rank = rank_str.index(card_str[1])
    elif len(card_str) == 3:
        suit = suit_str.index(card_str[0])
        rank = int(card_str[1:])
    else:
        raise ValueError(f"Invalid card string: {card_str}")
    return (rank, suit)

# Straight potential calculation
def calc_straight_potential(ranks):
    if not ranks:
        return 0.0
    # Count consecutive ranks
    unique_ranks = sorted(set(ranks))
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i-1] + 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    # Also check for wheel straight (A-2-3-4-5)
    if 14 in unique_ranks and {2, 3, 4, 5} & set(unique_ranks):
        wheel_count = 1 + sum(1 for r in [2, 3, 4, 5] if r in unique_ranks)
        max_consecutive = max(max_consecutive, wheel_count)
    return min(max_consecutive / 5.0, 1.0)

# Flush potential calculation
def calc_flush_potential(suits):
    if not suits:
        return 0.0
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    return min(max(suit_counts.values()) / 5.0, 1.0)

def street_to_num(street):
    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    return street_map.get(street, -1)

def get_last_opponent_action(action_histories, my_uuid):
    for street in reversed(['preflop', 'flop', 'turn', 'river']):
        round_actions = action_histories.get(street, [])
        for action in reversed(round_actions):
            if action['uuid'] != my_uuid:
                return action, street
    return None, None

def calculate_bets(action_histories, my_uuid):
    my_bet = 0
    opponent_bet = 0
    streets = ['preflop', 'flop', 'turn', 'river']

    for street in streets:
        if street not in action_histories:
            continue

        for action in action_histories[street]:
            if 'paid' in action:
                uuid = action['uuid']
                amount_paid = action['paid']
                if uuid == my_uuid:
                    my_bet += amount_paid
                else:
                    opponent_bet += amount_paid
            elif 'amount' in action and (action['action'] == 'SMALLBLIND' or action['action'] == 'BIGBLIND'):
                uuid = action['uuid']
                amount = action['amount']
                if uuid == my_uuid:
                    my_bet += amount
                else:
                    opponent_bet += amount
    return my_bet, opponent_bet

def encode_action_histories(all_action_histories, player_id, window_size=10):
    streets = ['preflop', 'flop', 'turn', 'river']
    features = []

    # Compute per-street raise frequencies
    for street in streets:
        # Player stats for this street
        player_raises = sum(
            1 for hist in all_action_histories
            for action in hist.get(street, [])
            if action['uuid'] == player_id and action['action'] == 'RAISE'
        )
        player_actions = sum(
            1 for hist in all_action_histories
            for action in hist.get(street, [])
            if action['uuid'] == player_id
        )
        features.append(player_raises / player_actions if player_actions > 0 else 0)

        # Opponent stats for this street
        opp_raises = sum(
            1 for hist in all_action_histories
            for action in hist.get(street, [])
            if action['uuid'] != player_id and action['action'] == 'RAISE'
        )
        opp_actions = sum(
            1 for hist in all_action_histories
            for action in hist.get(street, [])
            if action['uuid'] != player_id
        )
        features.append(opp_raises / opp_actions if opp_actions > 0 else 0)

    # Recent opponent raises (across all streets, last window_size hands)
    recent_opp_raises = sum(
        1 for hist in all_action_histories[-window_size:]
        for street in streets
        for action in hist.get(street, [])
        if action['uuid'] != player_id and action['action'] == 'RAISE'
    )
    features.append(math.tanh(recent_opp_raises / window_size))

    # action_map = {'SMALLBLIND': 0, 'BIGBLIND': 0, 'CALL': 0, 'RAISE': 1}
    is_opp_last_raise = -1
    is_last_raise = -1
    hist = all_action_histories[-1]
    for street in reversed(streets):  # Check latest street first
        for action in reversed(hist.get(street, [])):
            if action['uuid'] != player_id:
                is_opp_last_raise = 1 if action['action'] == 'RAISE' else 0
            else:
                is_last_raise = 1 if action['action'] == 'RAISE' else 0
            if is_opp_last_raise != -1 and is_last_raise != -1:
                break
        if is_opp_last_raise != -1 and is_last_raise != -1:
            break
    features.append(is_opp_last_raise if is_opp_last_raise != -1 else 0)
    features.append(is_last_raise if is_last_raise != -1 else 0)
    return features

def encode(hole_cards, community_cards, street: str, pot_size, stack, opponent_stack, round_count, is_small_blind, all_action_histories, player_id):
    hole_cards = list(map(card_to_num, hole_cards))
    community_cards = list(map(card_to_num, community_cards))

    # Extract hole cards
    card1, card2 = hole_cards
    rank1, suit1 = card1
    rank2, suit2 = card2

    # Basic hole card features
    high_card = max(rank1, rank2) / 12  # Normalized high card
    low_card = min(rank1, rank2) / 12   # Normalized low card
    high_suit = max(suit1, suit2) / 4  # Normalized high suit
    low_suit = min(suit1, suit2) / 4   # Normalized low suit
    suited = 1.0 if suit1 == suit2 else 0.0  # Whether cards are suited
    pair = 1.0 if rank1 == rank2 else 0.0    # Whether cards are a pair

    # Community cards features
    num_community = len(community_cards) / 5.0  # Normalize number of community cards

    # Process community cards
    community_ranks = [r for r, _ in community_cards]
    community_suits = [s for _, s in community_cards]

    # Count rank occurrences in community cards
    rank_counts_community = {}
    for rank in community_ranks:
        rank_counts_community[rank] = rank_counts_community.get(rank, 0) + 1

    # Pairs and trips in community cards
    pairs_community = min(sum(1 for count in rank_counts_community.values() if count == 2), 2) / 2.0
    trips_community = min(sum(1 for count in rank_counts_community.values() if count >= 3), 1) / 1.0

    # Process all cards
    all_ranks = [rank1, rank2] + community_ranks
    all_suits = [suit1, suit2] + community_suits

    # Count rank occurrences in all cards
    rank_counts_all = {}
    for rank in all_ranks:
        rank_counts_all[rank] = rank_counts_all.get(rank, 0) + 1

    # Pairs and trips in all cards
    pairs_all = min(sum(1 for count in rank_counts_all.values() if count == 2), 2) / 2.0
    trips_all = min(sum(1 for count in rank_counts_all.values() if count >= 3), 1) / 1.0

    # Calculate straight and flush potentials
    straight_potential_community = calc_straight_potential(community_ranks)
    flush_potential_community = calc_flush_potential(community_suits)
    straight_potential_all = calc_straight_potential(all_ranks)
    flush_potential_all = calc_flush_potential(all_suits)

    # Bettings
    my_bet, opponent_bet = calculate_bets(action_histories=all_action_histories[-1], my_uuid=player_id)

    # Combine all features
    features = [
        # hole cards
        high_card,
        low_card,
        suited,
        pair,
        high_suit,
        low_suit,
        # community
        num_community,
        pairs_community,
        trips_community,
        straight_potential_community,
        flush_potential_community,
        # all
        pairs_all,
        trips_all,
        straight_potential_all,
        flush_potential_all,
        # others,
        street_to_num(street) / 3,
        stack / 1000,
        opponent_stack / 1000,
        round_count / 500,
        is_small_blind,
        my_bet / 1990,
        opponent_bet / 1990,
    ] + encode_action_histories(all_action_histories, player_id)

    return features
