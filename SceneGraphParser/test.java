import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;
import java.io.*;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import java.util.logging.Level;
import java.util.logging.Logger;


public class test{
    public static void main(String[] args){
        // FileOutputStream fos = new FileOutputStream("test.json",true);
        // PrintStream out = new PrintStream(fos);
        Logger logger = Logger.getGlobal();
        try {
            PrintStream out = new PrintStream("test.json");  
            System.setOut(out);
        } catch (FileNotFoundException fnf) {
            // TODO Auto-generated catch block
            fnf.printStackTrace();
        } 
        RuleBasedParser parser = new RuleBasedParser();
        String output = "[";

        String jsonStr = "";
        try {
            File jsonFile = new File("../Charades_v1_test.json");
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile),"utf-8");
            int ch = 0;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read()) != -1) {
                sb.append((char) ch);
            }
            fileReader.close();
            reader.close();
            jsonStr = sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
        }
        JSONObject jobj = JSON.parseObject(jsonStr);
        // System.out.println(jobj); 
        JSONArray text = jobj.getJSONArray("description"); 
        JSONArray text_origin = jobj.getJSONArray("origin_description"); 
        String des_out = "[";
        for (int i = 0 ; i < text.size();i++){
            if (i % 100 == 1){
                logger.info(""+(i-1));
            }
            JSONArray des = (JSONArray)text.get(i);
            JSONArray des_origin = (JSONArray)text_origin.get(i);
            des_out = des_out + "[";
            for (int j = 0 ; j < des.size();j++){
                String sent  = (String)des.get(j);
                String sent_origin  = (String)des_origin.get(j);
                SceneGraph sg = parser.parse(sent);
                des_out = des_out + sg.toJSON(0,"unknown",sent_origin) + ",";
            }
            // System.out.println(des_out); 
            des_out = des_out.substring(0,des_out.length()-1) + "],";
        }
        des_out = des_out.substring(0,des_out.length()-1) + "]";
        System.out.println(des_out); 
        //System.out.println(sg.toJSON()); 
        logger.info("Finished");
        
    }
}


// public class OperateFile {

//     /**
//      * 从文件中读取信息，并转换为相应对象
//      * @return
//      */
    // public static FileDto readFileDto(){
    //     File file = new File("../Charades_v1_test.json");
    //     if(!file.exists()){
    //         return null;
    //     }
    //     FileDto dto = null;
    //     BufferedReader br = null;
    //     try {
    //         br = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));
    //         StringBuilder sb = new StringBuilder();
    //         String line;
    //         while((line = br.readLine()) != null){
    //             sb.append(line);
    //         }
    //         dto = JSONObject.parseObject(sb.toString(),FileDto.class);
    //         br.close();
    //     } catch (UnsupportedEncodingException | FileNotFoundException e) {
    //         e.printStackTrace();
    //     } catch (IOException e) {
    //         e.printStackTrace();
    //     }
    //     return dto;
    // }

//     /**
//      * 将对象写入缓存文件中
//      * @author 
//      */
//     public static void writeFileDto(FileDto dto){
//         File file = new File("../Charades_test_sg_pre.json");
//         if(!file.exists()){
//             try {
//                 file.createNewFile();
//             } catch (IOException e) {
//                 e.printStackTrace();
//             }
//         }
//         try {
//             String jsonStr = JSON.toJSONString(dto);
//             BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),"UTF-8"));
//             bw.write(jsonStr);
//             bw.flush();
//             bw.close();
//         } catch (UnsupportedEncodingException | FileNotFoundException e) {
//             e.printStackTrace();
//         }catch (IOException e) {
//             e.printStackTrace();
//         }
//     }
// }