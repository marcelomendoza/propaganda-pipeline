Propaganda Techniques
=====================

This page describes the 23 persuasion techniques detected by the pipeline.
Each entry lists the technique name, its internal runner identifier, and a
description based on the annotation guidelines from the JRC Technical Report
*"Annotation Guidelines for Propaganda Techniques"* (JRC132862).

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Technique
     - Runner
     - Description
   * - **Repetition**
     - ``RepetitionRunner``
     - Repeating the same message over and over again so that the audience will eventually accept it.
   * - **Exaggeration / Minimisation**
     - ``ExaggerationRunner``
     - Either representing something in an excessive manner, or making something seem less important or smaller than it really is.
   * - **Obfuscation**
     - ``ObfuscationRunner``
     - Using words which are deliberately not clear so that the audience may have its own interpretations.
   * - **Loaded Language**
     - ``LoadedLanguageRunner``
     - Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.
   * - **Whataboutism**
     - ``WhataboutismRunner``
     - A technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument.
   * - **Kairos**
     - ``KairosRunner``
     - Exploiting a specific moment in time by presenting arguments as uniquely urgent or timely, pressuring audiences to act immediately by suggesting that the opportunity or crisis is fleeting and will not return.
   * - **Conversation Killer**
     - ``ConversationKillerRunner``
     - Words or phrases (thought-terminating clichés) that discourage critical thought and meaningful discussion about a given topic.
   * - **Slippery Slope**
     - ``SlipperyRunner``
     - Arguing that one event will inevitably lead to increasingly negative consequences, assuming a direct causal chain without sufficient evidence between an initial action and an extreme endpoint.
   * - **Slogan**
     - ``SloganRunner``
     - A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.
   * - **Appeal to Values**
     - ``AppealToValuesRunner``
     - Connecting a message to deeply held beliefs or principles valued by the audience—such as patriotism, family, or justice—to encourage support without necessarily addressing logical arguments.
   * - **Red Herring**
     - ``RedHerringRunner``
     - Introducing irrelevant material to the issue being discussed, so that everyone's attention is diverted away from the points made.
   * - **Straw Man**
     - ``StrawmanRunner``
     - An opponent's proposition is substituted with a similar one which is then refuted in place of the original proposition.
   * - **Appeal to Fear / Prejudice**
     - ``FearPrejudiceRunner``
     - Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative.
   * - **Appeal to Authority**
     - ``AuthorityRunner``
     - Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered.
   * - **Bandwagon**
     - ``BandwagonRunner``
     - Attempting to persuade the target audience to join in and take the course of action because everyone else is taking the same action.
   * - **Casting Doubt**
     - ``CastingDoubtRunner``
     - Questioning the credibility or character of an opponent in order to undermine their argument without engaging with its substance.
   * - **Flag Waving**
     - ``FlagWavingRunner``
     - Playing on strong national feeling (or loyalty to any group, e.g. race, gender, political preference) to justify or promote an action or idea.
   * - **Smear / Poisoning the Well**
     - ``SmearPoisoningRunner``
     - An effort to damage or call into question someone's reputation by propounding negative propaganda before they have a chance to speak.
   * - **Tu Quoque**
     - ``TuQuoqueRunner``
     - Literally "you too": deflects criticism by pointing out similar or worse behavior by opponents rather than defending against the accusation, claiming moral inconsistency.
   * - **Guilt by Association**
     - ``GuiltByAssociationRunner``
     - Persuading an audience to disapprove of an action or idea by suggesting that it is popular with groups held in contempt by the target audience.
   * - **Name Calling / Labeling**
     - ``NameCallingRunner``
     - Applying derogatory or emotionally charged labels to individuals or groups in order to provoke fear, hatred, or contempt without engaging with their actual positions.
   * - **Causal Oversimplification**
     - ``CausalOversimplificationRunner``
     - Assuming a single cause or reason when there are actually multiple causes for an issue, including transferring blame to one person or group without investigating the complexities.
   * - **False Dilemma**
     - ``FalseDilemmaRunner``
     - Presenting two alternative options as the only possibilities when in fact more possibilities exist (black-and-white fallacy).
