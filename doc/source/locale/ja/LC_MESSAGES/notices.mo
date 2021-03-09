Þ          ´               L  £   M  H  ñ     :  à   C    $  Ë   ¥     q     y  *       ½     V     c  Q   l     ¾     ß	  ­   d
       	   !  w  +  Ï   £  v  s     ê  Ù   ï  õ  É    ¿     Î  !   Û    ý  µ        E     R  k   Z    Æ  á   H  ó   *          .   Data type, which is called "dtype", can be specified for NLCPy functions like NumPy ones. However, the current version of NLCPy supports only the following dtypes: Each dtype has character codes that identify it. In NLCPy, the character code 'q' and 'Q' are internally converted to 'l' and 'L', respectively. The dtypes and character codes other than described above are not supported yet. In addition, the current version does not support a structured data type, which contains above dtypes. Example: Here is a list of restrictions which are common to NLCPy functions. Besides these restrictions, there are some individual restrictions. Please see also the item of "Restrictions" in the detailed description of each function. If the unsupported dtype appears in the parameter list or the return type for NLCPy function, *TypeError* occurs. In case a NumPy function treats float16 type internally, the corresponding NLCPy function treats it as float32. Similarly, int8, int16, uint8, or uint16 is treated as int32 or uint32 during calculations. In such case the return value of NLCPy differs from that of NumPy. NLCPy API is based on NumPy one. However, there are some differences due to performance reasons. For example, when NumPy function returns a scalar value, NLCPy function returns it as a 0-dimension array. Notices Notices and Restrictions NumPy functions run on an x86 Node (VH). On the other hand, most of NLCPy functions offload automatically input ndarrays to a Vector Engine (VE), and then run on the VE. So, as computational cost becomes smaller than the offloading cost, NLCPy performance decreases. In this case, please use NumPy. Please note that there are functions which can not even support above dtypes. For example, the complex version of :func:`nlcpy.mean()` does not support. Restrictions Results: This page describes notices and restrictions which are common to NLCPy functions. To reduce overhead between Vector Host and Vector Engine, NLCPy adopts the lazy evaluation, which means that values are not calculated until they are required. So the position of warnings where your Python script raised may not be accurate. For details, see :ref:`Lazy Evaluation <lazy>`. To use NLCPy in your Python scripts, the package ``nlcpy`` must be imported. For more details, see :ref:`Basic Usage <basic_usage>`. Vector Host (x86) supports denormal numbers, whereas Vector Engine does NOT support it. So, if denormal numbers are caluculated in NLCPy functions, they are rounded to zero. character code data-type Project-Id-Version: nlcpy 1.0.0b1
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-01-21 16:22+0900
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: ja
Language-Team: ja <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.9.0
 "dtype"ã¨å¼ã°ãããã¼ã¿åã¯ãNumPyã®é¢æ°ã¨åãããã«NLCPyã®é¢æ°ã«ãæå®ã§ãã¾ãããã ããNLCPyã®ç¾å¨ã®ãã¼ã¸ã§ã³ã¯ãæ¬¡ã®dtypeã®ã¿ããµãã¼ããã¾ãã ådtypeã«ã¯ããããè­å¥ããæå­ã³ã¼ããããã¾ãã NLCPyã§ã¯ãæå­ã³ã¼ã'q'ã¨'Q'ã¯åé¨ã§ãããã'l'ã¨'L'ã«å¤æããã¾ãã ä¸è¨ä»¥å¤ã®dtypeã¨æå­ã³ã¼ãã¯ã¾ã ãµãã¼ãããã¦ãã¾ããã ããã«ãç¾å¨ã®ãã¼ã¸ã§ã³ã¯ãä¸è¨ã®dtypeãå«ãæ§é åãã¼ã¿åããµãã¼ããã¦ãã¾ããã ä¾: NLCPyã®é¢æ°ã«å±éããå¶éã®ãªã¹ããæ¬¡ã«ç¤ºãã¾ãããããã®å¶éã«å ãã¦ãããã¤ãã®åå¥ã®å¶éãããã¾ããåæ©è½ã®è©³ç´°èª¬æã®"å¶éäºé "ããè¦§ãã ããã ãµãã¼ãããã¦ããªãdtypeããã©ã¡ã¼ã¿ãªã¹ãã¾ãã¯NLCPyã®é¢æ°ã®æ»ãå¤ã®åã«è¡¨ç¤ºãããå ´åã*TypeError* ãçºçãã¾ããNumPyã®é¢æ°ãfloat16ã¿ã¤ããåé¨çã«å¦çããå ´åãå¯¾å¿ããNLCPyã®é¢æ°ã¯ãããfloat32ã¨ãã¦æ±ãã¾ããåæ§ã«ãint8ãint16ãuint8ãã¾ãã¯uint16ã¯ãè¨ç®ä¸­ã«int32ã¾ãã¯uint32ã¨ãã¦æ±ããã¾ãããã®ãããªå ´åãNLCPyã®æ»ãå¤ã¯NumPyã®æ»ãå¤ã¨ã¯ç°ãªãã¾ãã NLCPyã®APIã¯NumPyã«åºã¥ãã¦ãã¾ãããã ããããã©ã¼ãã³ã¹ä¸ã®çç±ã«ãããããã¤ãã®éããããã¾ãããã¨ãã°ãNumPyã®é¢æ°ãã¹ã«ã©ã¼å¤ãè¿ãå ´åãNLCPyã®é¢æ°ã¯ããã0æ¬¡åéåã¨ãã¦è¿ãã¾ãã æ³¨æäºé  æ³¨æäºé ããã³å¶éäºé  NumPyã®é¢æ°ã¯x86ãã¼ãï¼VHï¼ã§å®è¡ããã¾ãã ä¸æ¹ãã»ã¨ãã©ã®NLCPyã®é¢æ°ã¯ãèªåçã«ndarrayãVector Engineï¼VEï¼ã«ãªãã­ã¼ããã¦ãããVEã§å®è¡ãã¾ãããããã£ã¦ãè¨ç®ã³ã¹ãããªãã­ã¼ãã®ã³ã¹ãã¨æ¯ã¹ã¦å°ãããªãã»ã©ãNLCPyã®æ§è½ã¯ä½ä¸ãã¾ãããã®ãããªå ´åã¯ãNumPyãä½¿ç¨ãã¦ãã ããã ä¸è¨ã®dtypeããµãã¼ãã§ããªãé¢æ°ããããã¨ã«æ³¨æãã¦ãã ããã ãã¨ãã°ãè¤ç´ æ°åã® :func:`nlcpy.mean()` ã¯ãµãã¼ããã¦ãã¾ããã å¶éäºé  çµæ: ãã®ãã¼ã¸ã§ã¯ãNLCPyæ©è½ã«å±éããæ³¨æäºé ã¨å¶éäºé ã«ã¤ãã¦èª¬æãã¾ãã Vector Hostã¨Vector Engineã®éã®ãªã¼ãã¼ããããåæ¸ããããã«ãNLCPyã¯éå»¶è©ä¾¡ãæ¡ç¨ãã¦ãã¾ããã¤ã¾ããå¤ã¯å¿è¦ã«ãªãã¾ã§è¨ç®ããã¾ããã ãã®ãããPythonã¹ã¯ãªãããçºçããè­¦åã®ä½ç½®ãæ­£ç¢ºã§ãªãå ´åãããã¾ããè©³ç´°ã«ã¤ãã¦ã¯ã :ref:`éå»¶è©ä¾¡ <lazy>` ãåç§ãã¦ãã ããã Pythonã¹ã¯ãªããã§NLCPyãä½¿ç¨ããã«ã¯ãããã±ã¼ã¸ ``nlcpy`` ãã¤ã³ãã¼ãããå¿è¦ãããã¾ããè©³ç´°ã«ã¤ãã¦ã¯ã :ref:`åºæ¬çãªä½¿ç¨æ³ <basic_usage>` ãåç§ãã¦ãã ããã Vector Hostï¼x86ï¼ã¯éæ­£è¦åæ°ããµãã¼ããã¦ãã¾ãããVector Engineã¯ãµãã¼ããã¦ãã¾ããã ãããã£ã¦ãéæ­£è¦åæ°ãNLCPyã®é¢æ°ã§è¨ç®ãããå ´åããããã¯ã¼ã­ã«ä¸¸ãããã¾ãã æå­ã³ã¼ã ãã¼ã¿ã®ç¨®é¡ 