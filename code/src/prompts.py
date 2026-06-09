
import numpy as np
import torch

# Data type instructions for universal use
ANCHOR_DATA_INSTRUCTION = "This is the anchor data representing the average historical pattern or baseline behavior computed from past observations."
CURRENT_DATA_INSTRUCTION = "This is the current data representing the most recent actual measurements from the sensor network that captures the real-time state."
TARGET_DATA_INSTRUCTION = "This is the target anchor data representing the ground truth output corresponding to the anchor input, which serves as a reference for learning the prediction task."

PROMPTS = {
    "PEMS03": """PEMS03 dataset contains traffic flow data collected from Caltrans Performance Measurement System (PeMS) District 3, covering the North Central Valley area of California. This dataset includes data from hundreds of sensors aggregated into 5-minute intervals.

### Domain Knowledge:
Traffic flow forecasting requires capturing both spatial dependencies (road network topology) and temporal dependencies (daily and weekly patterns) to predict future traffic conditions accurately. The model must understand how traffic patterns propagate through the sensor network and evolve over time.

### Task Instruction:
You are tasked with predicting future traffic flow values for all sensor nodes in the network. Given historical traffic observations from multiple sensors organized in a graph structure, your goal is to forecast the traffic conditions for the next time horizon. The prediction should account for:
1. Spatial correlations between connected road segments
2. Temporal patterns including daily rush hours and weekly trends
3. The propagation of traffic congestion through the network
4. Non-linear dynamics in traffic flow behavior

### Expected Output:
Generate accurate traffic flow predictions for all nodes in the network for the specified future time horizon, considering both spatial and temporal patterns.""",

    "PEMS04": """PEMS04 dataset consists of traffic flow data from Caltrans Performance Measurement System (PeMS) District 4, representing the San Francisco Bay Area. It is a widely used benchmark for spatial-temporal traffic forecasting models.

### Domain Knowledge:
Traffic forecasting in dense urban areas like the Bay Area involves modeling complex interactions between different road segments and accounting for rush hour patterns and non-linear temporal dynamics. The San Francisco Bay Area presents unique challenges due to its complex highway network, multiple traffic flow patterns, and high traffic density during peak hours.

### Task Instruction:
Your objective is to analyze historical traffic data and generate accurate forecasts for future traffic speed and volume across the entire sensor network. Specifically, you must:
1. Extract meaningful patterns from the historical time-series data
2. Model spatial dependencies between interconnected sensors using the graph structure
3. Capture temporal dependencies including hourly, daily, and weekly periodicities
4. Predict future traffic states while accounting for sudden changes or anomalies
5. Generate predictions that are consistent across both spatial and temporal dimensions

### Expected Output:
Produce precise traffic flow and speed predictions for the upcoming time steps across all sensor locations in the network, ensuring spatial and temporal consistency.""",

    "PEMS07": """PEMS07 dataset comprises traffic data from Caltrans Performance Measurement System (PeMS) District 7, covering the Los Angeles area. This dataset is characterized by a large and dense network of sensors monitoring one of the busiest traffic regions.

### Domain Knowledge:
Forecasting traffic in Los Angeles requires robust modeling of high-volume traffic flows, recurring congestion patterns, and the propagation of traffic waves across a large spatial graph. The LA area is known for severe traffic congestion, complex highway interchanges, and highly variable traffic patterns throughout the day.

### Task Instruction:
You must utilize the historical time-series traffic data and spatial correlations embedded in the sensor graph to predict accurate traffic flow values for the upcoming time steps. Your predictions should:
1. Account for the large-scale spatial network with hundreds of interconnected sensors
2. Capture recurring congestion patterns specific to Los Angeles traffic
3. Model the propagation of traffic waves and bottleneck effects
4. Handle high traffic volumes and sudden congestion formation
5. Incorporate long-range spatial dependencies across the extensive road network
6. Generate forecasts that are reliable even during peak congestion periods

### Expected Output:
Deliver accurate traffic flow predictions for all sensors in the network for the specified future time horizon, with particular attention to congestion propagation and high-volume traffic dynamics.""",

    "PEMS08": """PEMS08 dataset collects traffic flow information from Caltrans Performance Measurement System (PeMS) District 8, spanning the San Bernardino and Riverside counties. It provides granular traffic data useful for analyzing suburban and highway traffic patterns.

### Domain Knowledge:
Traffic prediction in this region involves understanding highway flow dynamics, identifying bottlenecks, and capturing long-term temporal trends in traffic movement. This area represents suburban and inter-city highway traffic, which exhibits different patterns compared to dense urban areas, including longer-distance commuting patterns and interstate traffic flows.

### Task Instruction:
Based on the sequence of historical traffic observations, your task is to predict the future traffic state for the entire sensor network. Your prediction model should:
1. Understand highway traffic dynamics including free-flow and congested states
2. Identify and predict bottleneck formation and resolution
3. Capture commuting patterns typical of suburban areas
4. Model both short-term fluctuations and long-term trends
5. Account for the spatial structure of highway networks with on-ramps and off-ramps
6. Generate predictions that reflect the characteristic traffic patterns of suburban highways

### Expected Output:
Generate comprehensive traffic state predictions for all network nodes across the future time horizon, accurately reflecting highway traffic dynamics and suburban commuting patterns."""
}
 