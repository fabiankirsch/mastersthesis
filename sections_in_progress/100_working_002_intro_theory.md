::: {.cell .markdown}


\cleardoublepage

# Introduction

In 1820 about 90% of the world population lived in extreme poverty, in 1950 about 60%, in 1980 about 45%, in 2000 about 30% and in 2015 already less than 10% [@bourguinon_inequality_2009]. Increasing automation since the industrial revolution and the resulting exponential increase in productivity and economic growth continues to move people out of poverty [@bolt_m_2014]. People can afford better shelter, food, medical care and education [@lutz_world_2017; @peltzman_mortality_2009], while the working hours keep decreasing [@feenstra_next_2015]. While at first simple mechanical processes in seldom changing environments where automated, today intelligent machines take over more complex processes and human and machine work are more integrated [@noble_forces_2017]. This introduces a new problem, because in this work setting humans and machines need to exchange a lot more information.  While machines can efficiently exchange lots of information through digital networks [@dixit_energy_2015],  humans are very slow at communicating information from their brain to a machine. They can type something into a keyboard, push buttons and more recently they can also tell a machine what to do [@sheridan_humanrobot_2016]. Constantly communicating with a machine interferes with the human’s actual task [@cook_15_2017]. In addition, communicating unconscious or continuously changing mental states [@finkelstein_distinction_1999] is not even possible.  A solution to this is that a machine collects data about the human through sensors and makes sense of this data itself. This way the machine can receive information from the human without the human having to tell the machine. The goal of this master's thesis is to gain more insight into machine learning methods that can help machines recognize human activity and states from sensory data. This knowledge can then help to build machine interfaces that are better adapted for human machine interaction.


\cleardoublepage

# Theory

## Machine learning in human factors

In human factors system designs are optimized for human well being and general system performance [@iea_definition_2019]. Machine interfaces are designed based on human's perceptual, cognitive and physical abilities and limitations. For example, concerning the perception it is important to know the colors or tones a human can typically distinguish. Regarding the cognitive abilities it might be important to know how many items can be stored in working memory and for how long can they be stored there, how long is the typical attention span or for how long can humans usually perform a specific task until they become tired. The physical abilities determine for example how precisely an object can be grasped or how much weight can be carried for a particular time period without fatigue. And to understand how quickly a human can react to a particular stimulus all three abilities need to be considered [@proctor_human_2018].

![A traditional representation of a human-machine system (Image source: @proctor_human_2018, p. 11).](figures/human_machine_system_proctor_zandt_2018.png){#fig:human_machine_system}

In the traditional human-computer system shown in @fig:human_machine_system the machine can only adapt to the human through actions the human performs on the machine's controls. This limits the overall system performance in several ways. If the human performs tasks other than communicating with the machine those tasks might be interfered with or paused [@cook_15_2017]. If the human wants to communicate a cognitive state to the machine, translating this cognitive state into an action introduces a delay. Also, a human can only communicate states or actions that the human is actually aware of. Finally, the amount of information a human can communicate to a machine is very limited and is a bottleneck in the information transmission from human to machine [@suchman_plans_1987;@tufte_envisioning_1990].

To increase the amount, timeliness and precision of the information transmission from the human to the machine the 'Control' element in @fig:human_machine_system can be replaced or extended with sensors. This way the machine can gather information about the human activities and cognitive states without requiring any 'Action' from the human. Modern cars for example give a warning if the seat belt is not fastened and the weight on the seat crosses a defined threshold [@conigliaro_seat_1989]. This is a simple example and the conditions for the warning are defined in the software or electrical circuits of the car and will likely work for most humans. However, when using sensors to recognize more complex behaviors or mental states like attention, emotions, preference or fatigue pre-defining the conditions for particular states or activities in software or hardware might be impossible. Also individual differences might be so large that the conditions would need to be adapted for every person. This is where machine learning can facilitate human-computer interaction. In machine learning algorithms learn the relationships from data collected through sensors of particular mental states or activities [@bishop_pattern_2006;@koza_automated_1996]. These relationships are represented in a model, which can then be used by the computer to recognize complex activities or mental states [@muller_machine_2008;@rani_empirical_2006;@rautaray_vision_2015]. Today various sensors and brain-imaging techniques are available to collect data about mental and physical states and activities [@jacob_commentary_2003;@mukhopadhyay_wearable_2015;@tan_brain-computer_2010].

## Machine intelligence

In machine learning a computer looks at data and learns the relationships in that data – it builds a model of that data [@bishop_pattern_2006]. Like the human brain builds a model of the world from data it receives through the sensory organs. A child in Northern Europe learns that something very light, green and round might be a grape, something heavier and red might be an apple, and something very long and yellow a banana. It learns this by perceiving the shape, color and weight of the fruits through the visual sense and touch combined with a person telling the child the name of the fruit – the label. If the child sees something long and yellow in the future, it will know that this is banana. The same a computer can do if provided with a list of weight, color, height and width of each object as the input data and the name of the object (the label) as the output. Having built a model of these objects the computer will be able to give the correct label of the object when given the weight, color, height and width. Both, the computer and the brain are cognitive systems: they learn from experience – build a model –  and thereby can modify their behavior [@hollnagel_joint_2005]. To understand better how intelligent machines can be integrated into human societies one needs to know what capabilities machines currently have and will likely have in the future.

How intelligent are machines compared to humans and how intelligent will they be? Among many other definitions intelligence is defined as a memory system that can make predictions [@hawkins_intelligence_2007]. The human brain stores information and makes predictions, which include classifications like the identification of a banana –  so humans are intelligent. This also applies to a computer that does machine learning – so  also computers are intelligent.  Humans have general intelligence because their cognitive ability allows them to solve general problems [@simon_human_2017] and to perform a large variety of tasks from navigating their own body, speaking and understanding text to driving a car [@deary_intelligence:_2001;@gray_neurobiology_2004;@mackintosh_iq_2011]. To a lesser degree also animals have this general intelligence [@reader_evolution_2011]. However, machines until now can only solve specific problems and perform specific tasks like either driving a car or playing chess or speaking and are considered to have narrow intelligence [@legg_universal_2007]. The dominant opinion in the machine learning community is that it is only a matter of time until machines will have general intelligence and even super intelligence, i.e. being more intelligent than humans [@bostrom_how_1998]. Machines with general intelligence will be more flexible in the tasks they can perform and be able to solve general problems.

For machines to develop general intelligence, will they also become or need to become conscious during this process? It seems that the primary function of consciousness is to integrate information that would otherwise be independent [@baars_conscious_2002;@seth_theories_2006], and might therefore be a key factor in general problem solving. One of the common perceptions of consciousness in humans is ‘the experience of self’ and ‘knowing that one knows’ [@farthing_psychology_1992]. Related to this is the notion of qualia (Latin for ‘what kind of’), which describes subjective individual experiences like the pain of a headache or the green color of a leaf. The materialistic world view holds that matter is the fundamental substance in nature [@novack_origins_1965]. That consciousness emerges from matter seems counter intuitive to many and is called the ‘hard problem of consciousness’ [@chalmers_facing_1995]. There are multiple scientific theories on consciousness [@crick_framework_2003;@dehaene_conscious_2006;@parvizi_consciousness_2001;@tononi_information_2004]. The integrated information theory (IIT) [@tononi_integrated_2016] follows a computational approach and is most helpful when trying to understand consciousness in machines. The IIT starts from the conscious experience, regarding it as the only thing that is actually certain – as René Descartes already stated in the 17th century: “I think, therefore I am” [@burns_scientific_2001]. Using brain imaging techniques like EEG, fMRI, or PET [@di_perri_measuring_2016] one can see what happens in the brain when a human has a conscious experience –  the ‘neural correlates of consciousness’ [@koch_quest_2004]. The IIT describes the underlying physical structures and processes of a consciousness experience in so called ‘postulates’. These postulates can then be used to determine if a physical system, including machines, has consciousness based on its structure and processes.

The physical structures on which humans and machines process information currently differ very much. Humans, who miss parts of the thalamocortical system, usually have a reduced conscious experience and reduced general intelligence [@tononi_consciousness_1998]. However, people can miss the entire cerebellum, but have a normal conscious experience. The neurons in the thalamocortical system are highly connected including feedback loops, while the neurons in the cerebellum show only very few connections. The thalamocortical system helps us to solve general problems, while the cerebellum is very specialized and needed for human's fine motor skills [@arshavsky_cerebellum_1983; @granger_models_2007]. Unlike the human thalamocortical sytem, today’s machines compute information mostly serially [@von_neumann_first_1945], doing one computation after the other and without integrating any of that information or feeding it back into the computation process. This might be an explanation for why machines today have narrow intelligence, but don’t yet show behavior that would be considered generally intelligent or suggesting a conscious experience. On specific tasks, where integration of information is not essential computers already outperform humans by far: the number of computations that an ordinary computer chip can do ( ~ 3GHz, i.e. 3000000000 computations per second) is many orders of magnitude higher than that of a single neuron (max 0.5 - 1KHz, i.e. 500 -  1000 action potentials per second), which allows them to process information much quicker than a human. Another key difference between brains and computers is that in brains the processing units (neurons), and memory  (connections between neurons) are in the same place, while in computers these are separate. The CPU is doing the computations while the data is only temporarily loaded into the CPU cache and is generally stored on a hard drive, known as the von Neumann architecture [@von_neumann_first_1945]. However, prototypes of brain-inspired architectures for computers are being developed [@boybat_neuromorphic_2018]. These integrate the processing and memory units, making the processing of data faster and less energy intensive and potentially also pave the way for architectures that allow for integration of information and with that for more generally intelligent and conscious machines.


## Core methods and approaches in machine learning {#sec:core_methods}

In this section several methods and approaches will be presented that are important in developing a machine learning pipeline. Fundamental methods and approaches are presented first, more specific ones last.

### Uncertainty and iteration

The process of developing a machine learning pipeline is iterative and the outcome is uncertain. In contrast, software development can be planned much better, because with enough experience the developers can tell if something can be built and also give an estimate how much time it will take them. In machine learning a set of tools is available, but a machine learning engineer never knows what information the data holds and if algorithms will be successful at fitting a model to the data. Finding a well fitting machine learning pipeline is therefore an iterative process, during which different methods and algorithms are combined and tested in different ways. 	

### Algorithms and models

An algorithm is an abstract and clearly defined way to solve a certain problem or take a decision [@rogers_jr_theory_1987]. Algorithms define how a software decides what to do given a certain input. Algorithms can also be simple and mechanical like a thermostat on a radiator. It always behaves the same way given the same input unless reprogrammed or rebuilt. In machine learning algorithms are not used to define specific behavior, but to fit a model to the observed relationships in data [@bishop_pattern_2006]. What decision is being taken in the future (in machine learning this is usually called inference) depends on what has been in the past. In this case the particular algorithm defines only what process is applied to learn the structures present in the data and how decisions in the eventual model are represented. During the process of fitting the model the algorithm adapts the parameters of the model so that the model will eventually reflect the relationships in the data and be helpful in decision making. This fitting process can also be initialized with certain parameters, called hyper-parameters. The hyper-parameters define the complexity and architecture of the eventual model as well as the flow of the fitting processes itself.

### Learning approaches

In supervised learning a model is built based on finding relationships between inputs and outputs [@russel_artificial_2010]. The algorithm receives a data set which contains a defined output for all inputs. The outputs can be either continuous, e.g. the length of a fruit or categorical, e.g. the label “banana”. The former case is called regression, the latter classification. Semi-supervised is a variant in which not for all inputs outputs are provided.

In unsupervised learning only inputs are provided to the algorithm, which then tries to find similarities among these and cluster them in some way [@hinton_reducing_2006]. For example, providing only width and weight of a set of 100 fruits composed of bananas, apples and grapes, the algorithm should group all data entries in 3 clusters corresponding to the 3 fruits. Another example would be to detect anomalies or outliers in a data set. The result would be a single cluster containing the normal observations with abnormal observations lying outside that cluster.

In reinforcement learning the algorithm is provided with one or multiple outcomes that should be reached in a particular process and with a range of possible input values needed for that process [@sutton_introduction_1998]. The algorithm continuously adapts the inputs based on the outcome – successful strategies are reinforced. This type of algorithms is used when a computer should learn a game like chess or find the most efficient configuration for a power plant that. In chess the outcome for an input (moving a chess piece) is only clear in the end of the game (win or loss). Depending on the type of power plant the efficiency of the power plant will show with a certain delay. Computer simulations of real word processes are a common way in reinforcement learning to gain insights into the optimal configurations more quickly [@silver_general_2018].

See @sec:ml_cheat_sheets in the appendix for which algorithms are contained within each of these learning approaches and how to find a suitable algorithm.

### Loss functions, optimizers and performance metrics

:::::: {.cell .code tags=["hide"]}
```python

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['figure.figsize'] = [10, 5]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(20, -60)

X, Y, Z = axes3d.get_test_data(0.01)
Z = Z + 0.01 * (Z**2).T
cmap=cm.plasma
surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                       linewidth=0, antialiased=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('model parameter a')
ax.set_ylabel('model parameter b')
ax.set_zlabel('loss')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.savefig('figures/loss_function.png', bbox_inches='tight')
```
::: {.cell .markdown}
Loss functions are used to reflect how well the model is fit to the data [@wald_statistical_1950]. For example when trying to fit a straight line (the line reflects the predicted values) to a cloud of points (this cloud reflects the actual values) the loss function could be the mean of all squared distances of all points to the line on a certain dimension, which is called the mean-squared-error loss function (see @tbl:loss_functions for details and more examples). If the line lies outside the point cloud the loss will be very high and lower if it lies within the point cloud. The minimum of the loss function is to be assumed the best fit for the model. Assuming the model has only two parameters, the loss function can be thought of as mountains. The altitude of the mountains reflects the loss, the north-south dimension reflects one parameter and the east-west dimension the other. One starts the search for the valley in some random location and then starts to move around trying to find the bottom of the lowest valley, which reflects the minimum of the loss and the best model fit (see [@fig:loss_function]).

\newpage

Type | Name | Function
--- | -----  | ------
Regression | Mean Squared Error (MSE)| $(\sum\limits_{i=1}^n (y_{i} - \hat{y_{i}})^2) * \frac{1}{n}$
Regression | Mean Absolute Error (MAE) | $(\sum\limits_{i=1}^n \|y_{i} - \hat{y_{i}}\|) * \frac{1}{n}$
Classification | Cross Entropy Loss | $-(y_{i} log(\hat{y_{i}}) + (1-y_{i}) log(1-\hat{y_{i}}))$

: Loss functions for regression and classification in a supervised learning approach. {#tbl:loss_functions}


![Loss as a function of model parameters a and b. The best model fit is assumed to be at the minimum of the loss (darkest in this figure).](figures/loss_function.png){#fig:loss_function}

Optimizers are a set of algorithms, that decide how to tune the parameters of the model based on the output from the loss function so that eventually the minimum of the loss function is found. This includes both the direction of change (positive or negative) of the parameter as well as the amount of change. In the mountain analogy the optimizer decides which way to go and how far to go before checking again the altitude. Of course the obvious direction is to go down the mountain to the valley. So a very simple optimizer could just check all the gradients of its current location and move along the steepest negative gradient until reaching the lowest point where all gradients are positive and there is no way down. But what if just ended up in a small pit and the valley is still a long way to go or if the valley behind the next mountain ridge is even lower than the current valley? In technical terms these are local minima and if the optimizer stops changing the model parameters here, the best model fit will not be found ^[An interactive visualization of different optimizers: https://emiliendupont.github.io/2018/01/24 /optimization-visualization/ (2019-06-06)]. Optimizers are therefore usually designed in more complex ways, so they also do bigger jumps and don’t easily get stuck. In practice models can have millions of parameters to tune, which makes this task a lot more difficult.

Sometimes additional performance metrics are used besides the loss function to evaluate the model after the fitting process. Different loss functions reward or punish different structures in the data like the presence of outliers. So checking the outputs of other loss functions can help in understanding how well the model will be able to solve the problem at hand. More importantly, some metrics (e.g. accuracy) that are intuitive to humans are not smooth and can therefore not be used as loss functions because the optimizing algorithms cannot optimize it. A proxy loss function is therefore used that the algorithms can optimize. The human however will still want to look at the intuitive metric afterwards to understand how well the model is fit to the data.

:::


::: {.cell .markdown}


### Over-fitting and generalizability

Optimizing the loss function too much also has a downside. If the algorithm can handle all the complexity in the data there is a danger that the random noise in the training data is also modeled. This means the model can perfectly predict the outputs in the training data even though there is some random variance in the data that is actually not predictable. When applying the model to new data the model will fail, because the random noise in the new data will be different then in training data. This is called over-fitting the model [@domingos_few_2012]. Therefore, a common practice in evaluating the performance of a machine learning pipeline is to use different data for testing the model than for training the model.

The models are trained  (i.e. model parameters are adjusted) using a training set. The model’s performance is then checked on a test set to make sure that the model did not over-fit on the training data. However, testing different algorithms, algorithm-architectures and hyper-parameters of algorithms the final model might be over-fitted to the test data as well. Therefore a validation set is kept, which will only be used once on the final model. The performance of the model on the validation data will then be an accurate reflection of how the model performs on new data.

To reduce over-fitting during training regularization can be used to keep the parameters within certain limits or do not allow them to change too much [@friedman_elements_2001]. In the mountains analogy this would mean one wants to reach the valley, but one does not want to descend into some deep crevice as this is too specific.

### Feature engineering {#sec:theory_feature_engineering}

It can be beneficial to make certain aspects of the data more salient by manually generating new input variables from the provided input variables (also called features) - this is called feature engineering [@domingos_few_2012]. This is a manual process and domain knowledge is helpful to know what aspect of the data are important to predict the outcome. For example to highlight interactions between certain variables, these can be multiplied with each other or pair-wise correlations for variables can be computed. Other examples are to re-code a categorical variable that contains more than 2 labels into a single variable for each category, called ‘one-hot encoding’ or ‘dummy variables’.  These new features can then be used along with the provided features for the training of the algorithm. Some algorithms are also capable of extracting these feature themselves not requiring any manual work. This is called representation learning (de Jesus Rubio, Angelov, & Pacheco, 2015).

### Scaling and normalization

For many algorithms to perform well it is essential that the features have a similar center and variance. For example if feature A ranges from 0.1 to 0.2 and features B ranges from 10 to 1000 the algorithm might mistakenly interpret feature B as a better predictor. Therefore features are usually scaled to have a similar center and spread. A common method is to scale variables that all values are either within -1 and 1 or 0 and 1. Another method is to normalize the features, i.e. each feature has a center of 0 and a variance of 1 [@friedman_elements_2001].


### Dimensionality reduction

In high dimensional spaces (usually more than 30  features) it is harder to find significance because the data becomes sparser. This also referred to as the curse of dimensionality [@verleysen_curse_2005]. There are two different approaches to solve this: feature selection and feature projection.

In feature selection the original features are used, but features that do not contribute significantly to a better model fit are removed [@friedman_elements_2001]. For example in linear regression, there is forward selection where features are added one after another starting with the strongest ones and each time it is checked if the model fit improved significantly. Once a feature does not improve the model significantly anymore the procedure is stopped and no new features are added. In backward selection all features are added in the beginning and then removed one by one until the model fit drops significantly.

In feature projection the high dimensional feature space is projected into a lower dimensional features space using methods like principal component analysis (PCA), linear discriminant analysis (LDA) or independent component analysis (ICA) [@friedman_elements_2001]. These methods condense the information present in the many features of the high dimensional space to fewer features, that are orthogonal to each other, i.e. don’t share information. Which of these methods to choose usually requires some data exploration and domain knowledge. There are also fully automated methods like autoencoders [@hinton_reducing_2006], which is a type of neural network.

### Offline vs online learning

Once a model has been built with some initial training data inferences can be made on new data. However, in some cases the relationships in the new data will change over time. The initially trained model will not be an accurate presentation anymore of the relationships present in the latest data and the performance of the model will get worse over time. In this case the model needs to be retrained or updated. In offline learning the models gets retrained once in a while and then this retrained model will be used for inference from then on until the another model gets trained including the latest data. In contrast, in online learning for every new incoming data point an inference will be made and the data point will be also be used to update the model immediately. This way the model continuously adapts to the data.

### Filtering noise in continuous signals

If the data is a continuous signal from a physical sensor it might include noise. Noise in the signal will make it harder for an algorithm to find the relevant patterns. The noise can be random jitter due to measurement error of the sensor and not hold any information. Noise can also be another signal that is present in the data, but not relevant for the current problem. Using filters the relevant signal can be emphasized or extracted and the noise can be reduced or removed.

### Storage types

If data should rather be stored in files or in a database depends on the use case. Databases allow to quickly read data, so if speed is the top priority a database might the better choice. However, databases also have a computational overhead that requires additional computational resources and they need be setup which takes time. Relational databases also require the data to come in a previously specified schema, which can be cumbersome to setup when dealing with raw data. Files are very flexible and require no set up and are therefore the default choice for doing research and building prototypes.

### Batch processing vs online processing

In batch processing new observations are first stored and only passed to the pipeline after a certain number of observations (n) has been collected or after a certain delay (d). If the output of the pipeline is only needed once a day, it can be cheaper with regard to computational resources because the machine running the pipeline can be turned on only once a day.

In online processing new observations are passed immediately to the machine learning pipeline once they are available. This setup is usually chosen if the output of the pipeline is needed as soon as possible.  Online processing can be thought of as a variant of batch processing with n=1 and d=0.

### Sub-sampling

Data from a set of different sensors might be collected with a sampling rate of 100 Hz. However, if only characteristics, like temperature, which change at a much slower pace are of interest a sampling rate of 1 Hz might be enough. Through sub-sampling, i.e. taking only every 100th data point the algorithm only needs to processes 1% of the data.

### Testing
In software development tests are written which can automatically verify the integrity of the code base. Unit tests are used to verify that single functions work as as expected and integration tests verify the output final output of an application. This way the code can be changed and new code can be added without the need for elaborate and time consuming human testing to ensure the correct functioning of the application. In machine learning testing is even more crucial because the outcome might be very complex and not easily verifiable by a human. Therefore unit tests should be written for essential parts of a machine learning pipeline to guarantee the correct processing of the data.

## General composition of a machine learning pipeline {#sec:composition_pipeline}

A machine learning pipeline accepts one or more input variables, called features, and provides one or more output variables. From the collection of data to inference the data moves through three different blocks in a machine learning pipeline: extract-transform-load (ETL), pre-processing and modeling (@fig:general_ml_pipeline). These blocks are composed of different layers. A layer holds a particular statistical or data processing method, which is applied to the data. Some layers are always in the same block, some can be in different blocks and many of the layers are optional. Also, the order of the layers can be different depending on the use case. Prior to the data entering the pipeline it needs to be generated.

![General machine learning pipeline composition](figures/pipeline_flow.png){#fig:general_ml_pipeline}

### Storage

If data is not currently processed by a processing unit it needs to temporarily be stored somewhere. All the storing within the pipeline happens in-memory. This means the data is held in the machines working memory (RAM). This type of memory allows very quick write and read access and is necessary to interact with the processing units (CPU or GPU). However, working memory has a limited capacity and if the power of the machine is cut all information in working memory is lost.

Permanent storage (SSD or HDD) is slower in reading and writing and therefore not suitable for interacting with the processing units. However, it preserves information even if left without power, which makes it a save place to store data and models. Also, it has usually a much higher capacity than the working memory and can therefore hold all the data that is currently not being processed.

### Data generation

Data can be generated through sensors that measure some physical quality or it can be generated virtually by a computer. Examples for physical qualities are light, temperature, pressure or movement - this includes buttons of a computer keyboard. Virtual data generation by a computer could be monitoring the behavior of a software, logging the behavior of users on a website or even simulating data or randomly generating it. The generated data can then either be passed to the ETL block (@sec:pipeline_etl) or passed to permanent storage.

### Extract, transform, load (ETL) {#sec:pipeline_etl}

In this block data from different sources is extracted, i.e. selecting the data subset that one is interested in. So not more data than necessary will be processed. Data from different sources is also merged so it can be passed to the pipeline as a single set of observations. The format of the data might need to be changed, e.g. from a dictionary-format to a columnar-format. Also the data types might need to be adapted: during data collection all data points might have been saved as strings even if the data is actually numeric. These data points will then be transformed to integer or float values. Importantly, the actual data is not changed in the ETL block, only its structure.

The ETL block is specific to every data source that should be processed by the pipeline. Therefore multiple ETL blocks can exists for example if data from multiple systems should be processed. The input to this block is received either directly from the data generators or from permanent storage. The output of this block can be passed to permanent storage or to the pre-processing block (@sec:pipeline_preprocssing).

### Pre-processing {#sec:pipeline_preprocssing}

The pre-processing block can be very small or very large. In rare cases it might not exist at all. It holds all the layers that change the data in some way, but not yet build a model of the data. In this block, the data might be cleaned from noise, observations with missing values be removed, new features (columns) be generated through various feature engineering techniques or features removed by applying dimension reduction techniques. The output of this block might look very different than the input and it could be that none of the values passed to this block is present in the output. The output of this block can be passed to permanent storage or to the modeling block (@sec:pipeline_modeling).

### Modeling{#sec:pipeline_modeling}

The modeling block can contain multiple layers as different machine learning models might be combined and the output of one model serves as the input to the next model (stacked models). Initially the data is used for learning. Once the models are fitted to the data, these models are then used for inference on new data points. The models can be held in working memory or also permanently stored. Next to the models this block produces the actual output of the pipeline, i.e. some classification or prediction.