Þ                         ì      í   T   p  f   Å  ;   ,  [   h  h   Ä  J   -     x       1     ?   Ñ  ?     u  Q     Ç  c   Y     ½  5   D  ]   z  b   Ø  B   ;     ~       <     ?   Ø  ?   	   Fixed a bug in :func:`matmul` that returned invalid results when input ndarrays (a and b) fulfill any of the following conditions: Fixed a bug in :func:`prof.print_run_stats` that might showed negative elapsed time. Fixed a bug in :func:`sort` that might cause a freeze of Python interpreter at the end of the program. Improved performance of :func:`arccos` and :func:`arctanh`. Improved performance of broadcasting operations from a scalar value to an :class:`ndarray`. Improved performance of random number generators by changing the number of threads to be execeted on VE. Improved performance of universal functions with multi-dimensional arrays. Performance Enhancements Problem Fixes What's new in Version 1.0.0b2 (December 25, 2020) a.flags.c_congituous is False and a.flags.f_contiguous is False b.flags.c_congituous is False and b.flags.f_contiguous is False Project-Id-Version: nlcpy 1.0.0
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-02-05 14:32+0900
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: ja
Language-Team: ja <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.9.0
 å¥åndarrayï¼aããã³bï¼ãæ¬¡ã®ããããã®æ¡ä»¶ãæºããå ´åã«ç¡å¹ãªçµæãè¿ã :func:`matmul` ã®ä¸å·åãä¿®æ­£ã è² ã®çµéæéãç¤ºãå¯è½æ§ããã :func:`prof.print_run_stats` ã®ä¸å·åãä¿®æ­£ã ãã­ã°ã©ã ã®æå¾ã§Pythonã¤ã³ã¿ã¼ããªã¿ã¼ãããªã¼ãºããå¯è½æ§ããã :func:`sort` ã®ä¸å·åãä¿®æ­£ã :func:`arccos` ã¨ :func:`arctanh` ã®æ§è½æ¹åã ã¹ã«ã©ã¼å¤ãã :class:`ndarray` ã¸ã®ãã­ã¼ãã­ã£ã¹ãæä½ã®æ§è½æ¹åã VEã§å®è¡ããã¹ã¬ããã®æ°ãå¤æ´ãããã¨ã«ãããä¹±æ°çæã®æ§è½æ¹åã å¤æ¬¡åéåã«ããã¦ããã¼ãµã«é¢æ°ã®æ§è½æ¹åã æ§è½æ¹å ä¸å·åä¿®æ­£ ãã¼ã¸ã§ã³1.0.0b2ã®æ´æ°äºé ï¼2020å¹´12æ25æ¥ï¼ a.flags.c_congituous is False and a.flags.f_contiguous is False b.flags.c_congituous is False and b.flags.f_contiguous is False 