# Data Card

## Required details

The dataset I used is called CollegeMsg from Stanford SNAP. I got it from the Stanford SNAP website. It has 59,836 records in total. Each line has three things: sender, recipient, and timestamp, which is the time it was sent. Some people contacted each other back and forth, like user 9 contacted user 569, and later user 569 contacted user 9. That back and forth is what I used to calculate the temporal relationship score. The raw file is saved as raw_contact_data.csv in the data folder. After I calculated the scores, I saved a new file called temporal_scored_contacts.csv.

## Limitations

The dataset only has user IDs and timestamps. It does not have any names or friend labels. So I had to guess who was a "relevant" contact by counting how many times they contacted each other. The data is also just from one university, so it might not work the same way on other apps. There are no real friend labels, so I made my own rule to decide who is relevant. This means the results are based on my rule and not on what people actually think. You can see this in the evaluation part of the project.

## Verification

I opened the dataset, changed the timestamps into real dates, and counted how many times each person contacted each contact. I also checked how recent the last contact was and how far apart the contacts were. Then I made a rule to mark some contacts as relevant and some as not relevant. I checked the numbers myself to make sure they matched the raw file. You can see all of this in raw_contact_data.csv and temporal_scored_contacts.csv inside the data folder.
