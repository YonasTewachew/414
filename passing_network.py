#!/usr/bin/env python3
"""
Passing Network Analysis Script
This script analyzes a single soccer match passing network using StatsBomb open-data.
It filters for one team’s successful passes, builds a directed graph,
computes centrality measures, and visualizes the passing network.

Usage:
    Ensure you have cloned the StatsBomb open-data repository as described above.
    Then run: python passing_network.py
"""

import json
import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def load_match_events(match_file):
    """Load the JSON events for a match."""
    with open(match_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    return events

def filter_team_passes(events, team_name):
    """
    Filter events to include only successful passes for a given team.
    A pass is considered successful if the 'pass' dictionary does not include an 'outcome' key.
    """
    team_passes = []
    for event in events:
        if event.get('type', {}).get('name') == 'Pass':
            if event.get('team', {}).get('name') == team_name:
                # A missing outcome indicates a successful pass.
                if 'outcome' not in event.get('pass', {}):
                    team_passes.append(event)
    return team_passes

def build_passing_network(team_passes):
    """
    Build a directed graph where:
    - Nodes represent individual players.
    - Edges represent successful passes between players.
      Edge weights indicate the number of passes from one player to another.
    """
    G = nx.DiGraph()
    for event in team_passes:
        passer = event.get('player', {}).get('name')
        recipient = event.get('pass', {}).get('recipient', {}).get('name')
        if passer and recipient:
            if G.has_edge(passer, recipient):
                G[passer][recipient]['weight'] += 1
            else:
                G.add_edge(passer, recipient, weight=1)
    return G

def compute_centrality_measures(G):
    """
    Compute centrality measures:
    - Degree Centrality: Number of unique connections.
    - Betweenness Centrality: Frequency as a bridge between other nodes.
    - PageRank: Influence based on both quantity and quality of connections.
    """
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
    pagerank = nx.pagerank(G, weight='weight')
    return degree_centrality, betweenness_centrality, pagerank

def plot_network(G, degree_centrality):
    """Visualize the passing network with node sizes scaled by degree centrality."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [5000 * degree_centrality[node] for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=10, edge_color='gray', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Passing Network Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("passing_network.png")
    plt.show()

def main():
    # Path to the StatsBomb events data (from the cloned repository)
    data_path = "open-data/data/events/"
    match_files = glob.glob(os.path.join(data_path, "*.json"))
    
    if not match_files:
        print("No match event files found. Please clone the StatsBomb open-data repository into the 'open-data' folder.")
        return

    # For this analysis, we select the first match file.
    match_file = match_files[0]
    print(f"Analyzing match file: {match_file}")

    events = load_match_events(match_file)
    
    # Assume the team of interest is the home team from the first event.
    home_team = events[0].get('team', {}).get('name', 'Unknown Team')
    print(f"Filtering passes for team: {home_team}")
    
    team_passes = filter_team_passes(events, home_team)
    print(f"Total successful passes for {home_team}: {len(team_passes)}")
    
    # Build the passing network graph
    G = build_passing_network(team_passes)
    print(f"Number of nodes (players): {G.number_of_nodes()}")
    print(f"Number of edges (pass connections): {G.number_of_edges()}")

    # Compute and display centrality measures
    degree_centrality, betweenness_centrality, pagerank = compute_centrality_measures(G)
    centrality_df = pd.DataFrame({
        'Degree Centrality': degree_centrality,
        'Betweenness Centrality': betweenness_centrality,
        'PageRank': pagerank
    })
    print("\nCentrality Measures:")
    print(centrality_df.sort_values(by='Degree Centrality', ascending=False))

    # Plot and save the passing network
    plot_network(G, degree_centrality)

if __name__ == "__main__":
    main()
