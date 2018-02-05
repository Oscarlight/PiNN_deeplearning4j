package dcmodeling;

import java.awt.Color;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class Plot {
	
	
	public static XYSeriesCollection createDataset(MultiDataSetIterator iter, ComputationGraph net) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		iter.reset();
		while(iter.hasNext()) {
		  MultiDataSet data = iter.next();
		  XYSeries s = new XYSeries(data.getFeatures(0).data().asDouble()[0]);
          INDArray[] features = data.getFeatures();
          double[] Vds = data.getFeatures(1).data().asDouble(); // 1 : Vds or 0: Vds, dependent on input
          INDArray predicate = net.outputSingle(features);
          double[] Id = predicate.data().asDouble();
          for (int j = 0; j < Vds.length; j++) {
			s.add(Vds[j], Id[j]);
          }
          dataset.addSeries(s);
		}
		return dataset;
	}
	
	// TODO: read file and output dataSet
	public static XYSeriesCollection createDataset(MultiDataSetIterator iter) {
		XYSeriesCollection dataset = new XYSeriesCollection();

		iter.reset();		
		while(iter.hasNext()) {
			MultiDataSet data = iter.next();
			XYSeries s = new XYSeries(data.getFeatures(0).data().asDouble()[0]);
			double[] Vds = data.getFeatures(1).data().asDouble();  // 1 : Vds or 0: Vds, dependent on input
			double[] Id = data.getLabels(0).data().asDouble();
			
			for (int j = 0; j < Vds.length; j++) {
				s.add(Vds[j], Id[j]);
			}
			dataset.addSeries(s);
		}
		return dataset;
	}	
	
    /**
     * Plot out the result.
     * @param function
     * @param x
     * @param y
     * @param predicted
     * @throws IOException 
     */
    public static void linePlot(MultiDataSetIterator iter, ComputationGraph net) {
        final XYSeriesCollection data = createDataset(iter);
        final XYSeriesCollection model = createDataset(iter, net);
        
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Title",      // chart title
                "Vds", // x axis label
                "Id",  // y axis label
                model, // data
                PlotOrientation.VERTICAL,
                true, // include legend
                true, // tooltips
                false // urls
        );
        
        XYPlot xyplot = chart.getXYPlot();
		
		xyplot.setDataset(0, data);
		XYLineAndShapeRenderer trainRender = new XYLineAndShapeRenderer(false, true);
		// xylineandshaperenderer.setSeriesPaint(2, Color.YELLOW);
		trainRender.setBasePaint(Color.RED);
		trainRender.setAutoPopulateSeriesPaint(false);
		xyplot.setRenderer(0, trainRender);
		
		xyplot.setDataset(1, model);
		XYLineAndShapeRenderer modelRender = new XYLineAndShapeRenderer(true, false);
		// xylineandshaperenderer.setSeriesPaint(2, Color.YELLOW);
		modelRender.setBasePaint(Color.BLUE);
		modelRender.setAutoPopulateSeriesPaint(false);
		xyplot.setRenderer(1, modelRender);
		
        final ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(new java.awt.Dimension(600, 1000));
        
        final JFrame frame = new JFrame();
        frame.add(panel);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }
    
    
	public static void main(String[] args) throws IOException {
		INDArray x = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
		INDArray y = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
		INDArray[] predicate = new INDArray[2];
		predicate[0] = y.add(2);
		predicate[1] = y.mul(2);
	}

}
