# Ethics Statement

## Required details

This project uses the CollegeMsg dataset from Stanford SNAP. The dataset contains records of who contacted who and the time of contact. No message content, names, or personal information is included. The dataset is publicly available and was used only for learning and research purposes in this project. User IDs in the dataset are anonymous numbers, so no real person can be identified from them. The temporal relationship score I built uses only timestamps, not any private text or content.

## Limitations

The model was tested with a relevance rule that overlaps with the score, so it mainly shows the score finds pairs with many interactions, not real relationship strength. The baseline ranks contacts randomly and is not the existing system, and simpler options like ranking by interaction count were not tested. Cold-start was not simulated, the 0.3 and 0.7 weighting was not evaluated, and results come from a single run with no significance tests. The weights, decay constants, and 0.5 threshold were fixed in advance and not tested for sensitivity, and the data covers only one source over 193 days. Most pairs have just one interaction, no external labels were available, and only 35% of relevant pairs were found at the 0.5 threshold.

## Verification

I documented the limitations in the data card and model card, and I also stated them in the project documentation. I kept the raw file as raw_contact_data.csv and the processed file as temporal_scored_contacts.csv so the work can be checked. I used the dataset only for this research project. I followed the same rule for marking contacts as relevant across all users, and I checked the numbers myself to make sure they matched the raw data. I also made sure no personal information was used or shared at any point.


