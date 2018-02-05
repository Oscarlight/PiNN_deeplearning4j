package dcmodeling;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.data.xy.XYSeries;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


//////////////////////////////////////
// This is the version of MLPClassifierLinear for the screencast
// This example can also be found as part of a large collection 
// of examples at https://github.com/deeplearning4j/dl4j-examples.git
// with instructions on configuring your environment here
// http://deeplearning4j.org/quickstart
// More information at http://skymind.io/
//////////////////////////////////////

public class Network {
	
	private int seed, batchSize,
	            numInputSig, numInputTanh, numOutput,
	            numHiddenSig, numHiddenTanh;
	private double learningRate;
	
	public Network(int seed, double learningRate, int nEpochs, int batchSize,
			       int numInputSig, int numInputTanh, 
			       int numOutput, int numHiddenSig, int numHiddenTanh) {
		this.seed = seed;
		this.learningRate = learningRate;
		this.batchSize = batchSize;
		this.numInputSig = numInputSig;
		this.numInputTanh = numInputTanh;
		this.numHiddenSig = numHiddenSig;
		this.numHiddenTanh = numHiddenTanh;
		this.numOutput = numOutput; // sig == tanh
	}
	
	/**
	 * Config the network
	 * @return
	 */
	private ComputationGraphConfiguration config() {
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(seed)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1) // iterations per minibatch
	            .rmsDecay(0.95)
//	            .regularization(true)
//	            .l2(0.001)
				.weightInit(WeightInit.UNIFORM)
				.biasInit(0.5)
				.graphBuilder()
				.addInputs("sig_input", "tanh_input")
				.addLayer("tanh_L1", 
						new DenseLayer.Builder().nIn(numInputTanh)
												.nOut(numHiddenTanh)												
												.biasInit(0)
												.biasLearningRate(0)
												.updater(Updater.RMSPROP)
												.activation(Activation.TANH)
												.build(),
						"tanh_input")				
				.addLayer("sig_L1",
						new DenseLayer.Builder().nIn(numInputSig + numHiddenTanh)
												.nOut(numHiddenSig)
												.updater(Updater.RMSPROP)
												.activation(Activation.SIGMOID)
												.build(),
						"sig_input", "tanh_L1")
				.addLayer("tanh_L2", 
						new DenseLayer.Builder().nIn(numHiddenTanh)
												.nOut(numOutput)
												.biasInit(0)
												.biasLearningRate(0)
												.updater(Updater.RMSPROP)
												.activation(Activation.TANH)
												.build(),
						"tanh_L1")
				.addLayer("sig_L2",
						new DenseLayer.Builder().nIn(numHiddenSig + numOutput)
												.nOut(numOutput)
												.updater(Updater.RMSPROP)
												.activation(Activation.SIGMOID)
												.build(),
						"sig_L1", "tanh_L2")
				.addVertex("gating", 
						new ElementWiseVertex(Op.Product),
						"tanh_L2", "sig_L2")
				.addLayer("out_L", 
						new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
						                       .build(), 
						"gating")
				.setOutputs("out_L")
				.pretrain(false)
				.backprop(true)
				.build();
				
		return conf;
	}
	
	/**
	 * Read in the data from CSV file.
	 * @param dir
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	private MultiDataSetIterator readData(String dir, int batchSize) throws IOException, InterruptedException {
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dir)));
        return new  RecordReaderMultiDataSetIterator.Builder(batchSize)
        			.addReader("myReader", rr)
        			.addInput("myReader", 0, numInputSig - 1) // input one for Sig subnet
        			.addInput("myReader", numInputSig, numInputTanh + numInputSig - 1) // input two for Tanh subnet
        			.addOutput("myReader", numInputTanh + numInputSig, numInputTanh + numInputSig + numOutput - 1)
        			.build();
	}
	
    
    public static void main(String[] args) throws Exception{
    	// Random number generator seed, for reproducability
    	int seed = 123;
        double learningRate = 0.05;
        //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
        int batchSize = 21;
        // int batchSize = 51; // for the ease of plotting, put it as the same as period.
        // Number of epochs (full passes of the data)
        int nEpochs = 4000;
        // NN structure
        int numInputSig = 1;
        int numInputTanh = 1;
        int numOutput = 1;
        int numHiddenSig = 2;
        int numHiddenTanh = 3;
        
        Network pinn = new Network(seed, learningRate, nEpochs, batchSize,
			       			 	   numInputSig, numInputTanh, 
			       			 	   numOutput, numHiddenSig, numHiddenTanh);
        
        ComputationGraphConfiguration conf = pinn.config();
//        System.out.println(conf.toJson());
        // Vds, Vgs, Id 
        MultiDataSetIterator trainIter = pinn.readData("data/wse2_train.csv", batchSize);
        MultiDataSetIterator testIter = pinn.readData("data/wse2_test.csv", 42);
        // Vgs, Vds, Id 
//        MultiDataSetIterator trainIter = pinn.readData("data/p-Thin_TFET_train.csv");
//        MultiDataSetIterator testIter = pinn.readData("data/p-Thin_TFET_test.csv");
//        Plot.linePlot(trainIter);
        
        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        
        for (int n = 0; n < nEpochs; n++){
        	trainIter.reset();
        	while (trainIter.hasNext()) {
        		MultiDataSet data = trainIter.next();
        		INDArray[] features = data.getFeatures();
        		INDArray[] labels = data.getLabels();
        		net.fit(features, labels);
        	}
        	if (n % 1000 == 0) {
        		System.out.println(n + ": training score: " + net.score());
        		System.out.println(n + ": testing score: " + net.score(testIter.next()));
        		testIter.reset();
        	}
        }
        
        /* Plot out result */
        Plot.linePlot(trainIter, net);
        Plot.linePlot(testIter, net);
        
        /* Save the model */
        File locationToSave = new File("model/wse2-pinn.zip");       			//Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                     //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

    }
}