import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.Properties;

import javax.imageio.ImageIO;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer09;
import org.apache.flink.streaming.util.serialization.JSONDeserializationSchema;
import org.apache.flink.streaming.util.serialization.SimpleStringSchema;
import org.apache.flink.util.Collector;
import org.apache.flink.streaming.util.serialization.AbstractDeserializationSchema;

import com.fasterxml.jackson.databind.node.ObjectNode;
//import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.node.ObjectNode;

@SuppressWarnings("serial")
public class LrtoHrMapper extends AbstractDeserializationSchema<byte[]>{
	
	

	public static void main(String[] args) throws Exception {



		// get the execution environment
		final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		
		Properties properties = new Properties();
		properties.setProperty("bootstrap.servers", "localhost:9092");
		// only required for Kafka 0.8
		properties.setProperty("zookeeper.connect", "localhost:2181");
		properties.setProperty("group.id", "UserNameVideoName");
		FlinkKafkaConsumer09<byte[]> consumer = new FlinkKafkaConsumer09<>("UserNameVideoName",new LrtoHrMapper(), properties);
		consumer.setStartFromEarliest();
		DataStream<byte[]> stream = env
			.addSource(consumer);

		// count the frames high-resoluted
		DataStream<Integer> count = stream.map(new MapFunction<byte[], Integer>() {
		    @Override
		    public Integer map(byte[] node) throws Exception {
	            int num = 0;
		    	for(int i=0; i< node.length;i+=1){
		    		if(node[i]!= 35){
		    			num*=10;
		    			num+=node[i]-48;
		    		}
		    		else{
		    			break;
		    		}
		    	}
		    	
		    	//nod
		    	// write the frame into a file
		    	//System.out.println("------------incoming frame no is"+node.length+"-------");
		    	//System.out.println("------------incoming frame no is"+node.get("value").asText().length() +"-------");
		    	//String[] split = value.split("#", 2);
		    	String fileName = "UserNameVideoName_" + Integer.toString(num)+ ".txt";
		    	try (FileOutputStream fos = new FileOutputStream(fileName)) {
		    		   fos.write(node);
		    		   fos.close();
		    		}
		    /*	byte[] imageInByte= new BigInteger(value, 16).toByteArray();
		    	 InputStream in = new ByteArrayInputStream(imageInByte);
		    	 BufferedImage bImageFromConvert = ImageIO.read(in);
		    	 ImageIO.write(bImageFromConvert, arg1, arg2) */
		    	
		    	String param = "--input "+fileName+ " --topic_id UserNameVideoName --sizeX "
		    	  		+ args[2] + " --sizeY "+ args[3];
		    	System.out.println(param);
		    	try{
		    	Process p = Runtime.getRuntime().exec("python /home/ani2404/Desktop/ee6761cloud/KafkaProducer.py "+ param);
	
		    	int exitVal = p.waitFor();
	            System.out.println("Process exitValue: " + exitVal);
		    	}
		    	catch (Throwable t)
		          {
		    		System.out.println("Did not wait");
		            //t.printStackTrace();
		          }
		    	
		    	
		        return 1;
		    }
		   });

		count.print();
		
		
		env.execute("High resolution video");
	}

	@Override
	public byte[] deserialize(byte[] arg0) throws IOException {
		// TODO Auto-generated method stub
		return arg0;
	}


}