
# From our testing dataset

```
i feel a colon cancer meltdown coming on i probably wouldnt be worrying so much about colon cancer if i hadnt spent nearly 24 years of my life being fed ultra processed foods and then going on to cook ultra processed foods for myself i wish i was more self informed about how bad this food was for me im just now starting to figure out whats good for my colon even though i thought i was doing things right eating frozen vegan patties which may be even worse now that i think about it i was worried about processed meat only and not processed meals in general ive had hemorrhoid symptoms lasting over 2 months now itching and bright red spots bowel habits constantly change lot of gas bloating and nausea feeling hungry more than usual i had diarrhea for the first time in a month and a half last weekend no bowel movements at all for almost 4 days tldr im nearly 24 years old and have eaten processed food most of my life worried about colon cancer because of that but also really hoping i just have some chronic hemorrhoids and improving my diet moving forward
```
- Ground: Anxiety
- Predicted: Anxiety by all models

```
i hate life i want to take these tablets amp hope i never wake up but so scared that i do wake like the last time fuck it fingers crossed tonight is the night good bye you all so conflicted
```
- Ground: Suicidal
- Predicted: Suicidal by all models

```
i feel like my mental illness has completely overtaken my words actions decision making etc to the point that i am such an entirely different person its like i have rewired my brain to only think and act out of sadness or worry i overly guard my every action in fear i will ever even slightly inconvenience them its in every part of my life and i am lost i have no clue who i am anymore completely lost
```
- Ground: Depression
- Predicted: Depression by all models

```
i want to buy lunch
```
- Ground: Normal
- Predicted: Normal

```
tired on abilify so i was officially diagnosed 2 days ago as bipolar and my doctor has prescribed me 15mgs of abilify im about 59 and 110lbs and i feel like im a zombie today is day 2 of the meds and im exhausted yesterday i woke up feeling almost like i was hungover i felt nauseous dizzy and hot today i was a bit dizzy when waking up and after about 3 hours of jitteryness i just feel exhausted im at work in the back ready to pass out sleeping at any moment does this go away will this tiredness adjust as i adjust to the med
```
- Ground: Bipolar
- Predicted: Bipolar by all models (Note: Remove the phrase "as bipolar" in the first sentence and it still predicts correctly!)

# From outside datasets
[Mental_Health_Condition_Classification](https://www.kaggle.com/datasets/haideradnan77/mental-health-condition-classification)
- A similar dataset with the same classes (except doesn't have suicidal nor normal classes), but from differnet sources (i.e. used LLM to generate responses & does a permutation on sentences)
- The following examples then show how the model performs both in contexts outside of Reddit posts and also a separate set of labels.
```
I'm drowning in responsibilities, each one demanding my attention, yet I fear that no matter how hard I try, I may never truly conquer the mountain of tasks before me.
```
- Ground: Stress
- Predicted: Depression by all models

```
"My mind is a whirlwind of worry and fear, making even the simplest tasks feel insurmountable."

Another example:

"Every moment is tinged with unease, and I can't seem to find peace in my own thoughts."

A third example:

"The weight of my anxiety bears down on me like a heavy burden, making it hard to focus on anything but my racing thoughts."

A fourth example:

"I'm held captive by my thoughts, unable to escape the cycle of fear and self-doubt."

A fifth example:

"Anxiety grips me like a vice, making it difficult to
```
- Ground: Anxiety
- Predicted: Depression by XGB & LSTM, Stress by BERT

```
Every moment feels heavy with uncertainty, my mind a whirlwind of anxious thoughts that I can't seem to escape.

I'm constantly on edge, my body tense with fear, as if I'm bracing myself for the next disaster to strike.

I wake up in the middle of the night, my mind racing, my heart pounding, unable to shake the feeling of impending doom.

My thoughts are a chaotic storm, each one more frightening than the last, leaving me feeling trapped and helpless.
```
- Ground: Anxiety
- Predicted: Anxiety by XGB, Depression by LSTM, Stress by BERT

```
Amidst the constant barrage of demands and expectations, I struggle to find moments of peace and tranquility, leaving me feeling drained and unable to focus on what truly matters. Despite my relentless efforts to keep up, the escalating demands of my daily routine have left me feeling like a hamster on an endless wheel, spinning but getting nowhere.
```
- Ground: Stress
- Predicted: Depression by XGB & LSTM, Stress by BERT

```
Last week, I was engrossed in a creative project, brimming with inspiration and productivity. Now, I'm consumed by a deep sadness and an overwhelming sense of hopelessness.

Just a few hours ago, I was lost in a manic frenzy, filled with grand plans and limitless enthusiasm. But now, I'm crashing, exhausted and unable to focus on anything but the heaviness in my bones and the emptiness in my mind.

This morning, I woke up feeling normal, stable, and in control. But as the day wore on, the familiar clouds of depression began to gather.
```
- Ground: Bipolar
- Predicted Bipolar by XGB, Depression by LSTM & BERT

[Reddit Posts on Borderline Personality Disorder](https://www.kaggle.com/datasets/nourmekkijj/reddit-posts-on-borderline-personality-disorder)
- This is a dataset of reddit posts from a borderline-personality-disorder section.
- The following results show how the model performs in classifying personality disorder.
```
my mom is ubpd as an adult i now have this weird spider sense where i feel like i know when things are going to pop off or at the very least i know that shes not happypleased with our current communication levels before she fires off a text i can feel the tension and its a bit anxiety inducing on some level learned behavior from childhood

anyone else have this

how do you deal with it
```
- Ground: Personality Disorder
- Predicted: Stress by XGB, Normal by LSTM, Personality Disorder by BERT

```
so when i spiral i reach for the bottle otherwise i can be perfectly content and sober my spirals are often due to feelings of rejection crippling loneliness but also when engage in relationships its a lose lose
what im hoping for advice on
•is alcohol abuse and bpd common
•success stories in coping with this
•any tips and tricks

it is really impacting the way my friends and family are seeing meany help will be greatly appreciated
```
- Ground: Personality Disorder
- Predicted: Stress by all models
```
hi ive recently been told to go to a sun support group as in my initial assessment with a mental health practitioner he suggested that i may have bpd and they want to assess me for it but im a bit scared has anyone been to this group and able to give me some advice or what to expect this is all very new for me x
```
- Ground: Personality Disorder (but reads like stress)
- Predicted: Stress by XGB and BERT, normal by LSTM


