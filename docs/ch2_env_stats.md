# C-CyberBattleSim Chapter-2 Environment Statistics

## Dataset
- Dataset root: `cyberbattle/data/env_samples/syntethic_deployment_20_graphs_100_nodes`
- Model file preference: `network_CySecBERT.pkl`
- Number of analyzed environments: 20
- Split (from split.yaml):
  - train: 12 envs
  - validation: 4 envs
  - test: 4 envs

## Topology-Level
- Nodes per env: mean=100.00, std=0.00, min=100, max=100
- knows edges: mean=7235.60, std=696.04, density mean=0.7309
- access edges: mean=7964.85, std=1097.41, density mean=0.8045
- dos edges: mean=8291.20, std=1226.87, density mean=0.8375

## Node and Firewall
- has_data ratio mean: 0.7500
- visible ratio mean: 0.3095
- services per node (env means): mean=2.22
- vulnerabilities per node (env means): mean=33.29
- incoming firewall BLOCK ratio mean: 0.0000
- outgoing firewall BLOCK ratio mean: 0.0000

## Vulnerability Profile
- vulnerability instances per env: mean=3329.40
- unique vulnerability IDs per env: mean=558.10
- attack vectors: {'NETWORK': 58629, 'LOCAL': 4998, 'PHYSICAL': 203, 'ADJACENT_NETWORK': 2758}
- attack complexity: {'LOW': 53929, 'HIGH': 9368, 'MEDIUM': 3291}
- base severity: {'HIGH': 20583, 'CRITICAL': 6187, 'MEDIUM': 35471, 'LOW': 4347}
- privileges required: {'NoAccess': 38362, 'LocalUser': 13776, 'ROOT': 14450}
- base score (env means): mean=6.350
- exploitability score (env means): mean=3.416
- impact score (env means): mean=3.815
- success rate (env means): mean=0.9676

## Predicted Outcomes (Top 12)
- DenialOfService: 45317
- Exfiltration: 19201
- Collection: 19118
- PrivilegeEscalation: 17874
- LateralMove: 15589
- DefenseEvasion: 15297
- Execution: 12560
- Discovery: 12560
- Reconnaissance: 12560
- Persistence: 12560
- CredentialAccess: 3822

## Split Consistency
- training: count=12, nodes_mean=100.00, knows/access/dos edges mean=7488.67/8456.83/8844.17
- validation: count=4, nodes_mean=100.00, knows/access/dos edges mean=6927.50/7640.50/8171.25
- test: count=4, nodes_mean=100.00, knows/access/dos edges mean=6784.50/6813.25/6752.25

## Top Ports (frequency)
- 61092: 2
- 10124: 2
- 44449: 2
- 64792: 2
- 56927: 2
- 21914: 2
- 32114: 2
- 17017: 2
- 2508: 2
- 26577: 1
- 23347: 1
- 24786: 1
- 44493: 1
- 57648: 1
- 27658: 1
