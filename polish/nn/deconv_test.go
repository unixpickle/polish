package nn

import (
	"math"
	"runtime"
	"testing"
)

func TestDeconvLarge(t *testing.T) {
	runTest := func(t *testing.T) {
		c := &Deconv{
			OutDepth:   4,
			InDepth:    5,
			KernelSize: 3,
			Stride:     2,
			Weights: []float32{
				-0.1219822, 0.0415145, -0.0003108, 0.0420820, -0.0970473, -0.1313528, 0.1042072, -0.0170022, -0.0938565, -0.0299630, 0.0318583, 0.0244425, 0.0210479, 0.0111827, -0.0148932, 0.0557383, 0.1224927, 0.0547441, 0.1582459, 0.0499690, 0.1207057, -0.0196894, -0.0755690, 0.0689451, 0.0779862, 0.0873665, 0.1419234, -0.0543128, 0.1608845, 0.0582658, 0.0228303, -0.0477583, 0.0354382, 0.0951089, 0.0891034, -0.0037552, -0.1537833, 0.1476649, -0.0221033, -0.0759088, -0.0557675, -0.1296122, 0.1325820, 0.0835043, -0.1161792, 0.1155595, 0.0505537, -0.0784579, -0.0735167, 0.0492661, 0.1260416, -0.1229414, 0.1009175, -0.1287802, 0.1222984, -0.0440249, -0.0013464, -0.1196105, -0.1063310, -0.1411955, 0.0688984, 0.1312972, -0.0141451, -0.0510540, 0.0973860, -0.1332392, -0.1191129, 0.0066505, 0.1188149, 0.0877433, 0.1113341, 0.1222921, -0.1393880, -0.0432469, -0.0526133, -0.0949420, -0.1076849, -0.0002497, -0.1306648, 0.0294515, -0.0090034, 0.0040909, -0.1298766, 0.0008129, -0.0666431, 0.0601226, 0.0552047, -0.1309684, -0.0444079, -0.1611355, 0.1431194, 0.1094252, 0.1139870, 0.0102238, -0.1103540, -0.1296468, -0.0996154, -0.0453108, 0.1056267, 0.1260485, 0.1171038, -0.1362305, 0.1390443, 0.0645312, -0.1069985, -0.1441645, -0.1028010, -0.0422021, 0.0047795, -0.0863843, 0.0737386, 0.0269379, -0.0824751, 0.1281872, -0.1497540, 0.0671954, -0.1099316, 0.0260724, 0.0106858, -0.1560881, -0.1535428, 0.0284854, 0.1236318, -0.1221088, -0.1611681, 0.0809807, 0.0805877, 0.0553429, 0.0253775, 0.0030661, -0.0974576, -0.1127261, -0.0760133, -0.0775680, 0.1187243, -0.0476503, -0.1262008, 0.0253365, 0.0707496, -0.0859080, 0.1214689, -0.0510921, 0.0854513, 0.1506919, 0.0251312, -0.1434479, 0.1403285, -0.0385780, 0.1050188, 0.0446120, 0.0138543, 0.1392026, 0.0407732, 0.1158056, -0.1064162, -0.0432423, -0.0038336, -0.1366038, 0.0687647, -0.1084612, -0.1086784, -0.1020633, 0.0336938, 0.1003439, 0.0182202, 0.0090614, 0.1004127, -0.0765266, -0.1150280, -0.1253591, 0.1291900, -0.0042546, 0.0653943, 0.0769058, -0.1206319, -0.0232126, 0.1008501, -0.1636609, -0.0687802, -0.0380300,
			},
		}
		in := NewTensor(6, 7, 5)
		copy(in.Data, []float32{
			-1.2988282, 0.2510391, 1.0101794, 0.0262716, -0.7962795, -1.3709655, 1.5619336, -4.1645885, 1.3380519, -1.3223622, 1.3869857, 0.1907596, 1.5335323, -1.2222977, 0.1216181, 0.3406470, 0.1903207, 0.6818309, 0.5130429, 1.0867368, 0.1619944, 0.1881683, -1.0403576, -0.5780251, 0.6729466, -0.1694447, -1.8407322, -0.3128998, 1.2927204, -1.9052949, 0.3897138, 1.0135902, 0.3811399, 2.2074044, -0.5668194, -1.3145959, -0.5424687, -1.2134125, -0.4866764, 1.9015125, -0.5780223, 0.6818247, -1.5994284, -0.9151881, 0.1030166, -0.7112994, 1.0685599, -0.3549566, -2.1857874, -1.0917655, -0.6021930, 1.1224357, 1.6376072, -0.5959309, 1.0260910, -0.0712967, -1.2713999, -0.0322425, -0.1124378, 0.9655091, 0.8144102, 0.0382287, -0.1795858, -0.2953408, -0.4305271, -0.2038272, -0.1020874, -0.2738957, 0.3701357, 0.7156716, 0.0504618, -0.9757811, 0.7032495, 0.6709840, -1.0611930, -0.3453544, -0.3327364, 0.7522176, -0.1924587, -0.3480925, -0.3838927, 0.4166580, -1.2638218, 0.1913680, 1.5650450, -0.8434668, -0.6515325, 0.4179985, 0.9464355, -0.8318047, 1.0476940, 1.9159533, 0.5978024, -1.3885217, -1.3461010, -0.7859114, 0.9567471, -1.6975802, 1.7899201, 0.9705070, 0.4316317, 0.0558143, 2.8513019, 2.0602522, 1.3202075, -2.0741122, -1.8953130, 0.8939016, 0.9133815, -2.2644036, -0.6281854, 0.3507041, 0.5025447, 0.4345118, 0.0053923, 1.8058343, 0.7382588, 1.8524241, -0.0612551, 1.2026122, 1.7526712, -0.1822277, 0.6281086, 2.0601828, -0.1985361, 0.6348800, 0.6708742, -1.0700990, -0.4328031, 0.7890400, 0.6280078, 0.4541491, -0.7496076, -1.4483989, -2.1997216, -0.4227039, -1.5337064, -1.1435885, 1.1559694, -0.4253571, -0.6400719, 1.2148696, 0.2517402, -1.5043896, -1.8011912, 0.8911669, 0.0587563, -0.3249962, 0.6687999, -0.0895100, -0.6761006, 0.4636458, -0.5522307, 0.9506739, 1.1748210, -0.3795080, 0.0771183, -0.0588746, -0.2903397, -0.4915347, -2.0633969, -0.2064304, -0.9264892, -0.1987959, 1.7669750, 0.7308894, 0.0915315, -0.2743991, -0.9268026, 2.4203701, 0.1530970, -0.6694155, 0.4007120, -0.1968893, -0.1985747, -0.3613594, 0.4583485, 0.4560544, 1.2247474, 1.3021513, 2.0714045, 0.3739655, 0.1966641, 0.9291840, 1.6853676, 0.6274776, -0.0975672, -0.9863145, 0.7114818, -0.5865821, 0.7559505, -1.4822845, 0.0133979, 2.5776844, -0.1955276, -1.3894904, 0.0986306, -0.6527658, -1.0126148, -1.3212727, -0.2197127, 1.0792760, 0.4495734, -0.0663339, 0.6242186, 1.8274466, 0.2306639, -0.5027097, 0.7396556, -1.5373755,
		})

		expected := NewTensor(13, 15, 4)
		copy(expected.Data, []float32{
			-0.0408645, -0.0194694, -0.0549688, 0.1871940, 0.0514174, -0.0748684, -0.0438619, -0.1216055, 0.3125930, 0.0659977, -0.6144946, -0.8956703, 0.4279370, 0.7311863, -0.6516200, -0.8114839, -0.3169491, -0.3421381, -0.2645444, 0.3775117, 0.1075694, -0.1713427, 0.1732734, 0.5835122, -0.2944925, 0.3595108, 0.5658865, -0.1464769, -0.1874506, -0.1782442, 0.2206931, 0.1595043, 0.2606104, -0.0576293, 0.0272467, -0.1266818, 0.0329028, 0.0720025, -0.0784942, 0.0395116, 0.4079916, -0.3463828, -0.3584284, 0.1734509, -0.1036744, 0.1587522, -0.0813099, -0.5309020, -0.3750744, 0.1195588, 0.3434209, 0.0365922, 0.0399913, 0.0980617, 0.0818439, -0.1096016, 0.0406522, -0.3897277, 0.1348119, -0.1519297, -0.1381952, -0.1140959, -0.0012608, 0.1788205, -0.0825243, 0.1681010, -0.1225363, 0.1451142, 0.4118539, -0.0112780, -0.4082728, -0.7436250, 0.2451778, 0.0299881, 0.1339143, -0.2771372, -0.0481255, 0.1620539, 0.1381359, 0.7327967, -0.1967984, 0.0656770, -0.1629940, 0.1361714, -0.4514995, -0.1899078, -0.0127284, -0.2433596, -0.0452810, -0.0796590, -0.0621000, -0.0403041, 0.0944393, 0.3397935, -0.2647886, -0.1517983, 0.2041608, -0.1598594, 0.2064631, -0.0395846, 0.1811487, -0.0998886, 0.3401464, 0.6302477, -0.1539173, 0.1857012, -0.0742397, -0.0911694, 0.3262306, -0.6464262, 0.1654890, -0.0606451, -0.3769718, 0.2175172, -0.4513310, -0.1637520, 0.0750152, 0.3769193, -0.3711143, 0.3044261, 0.2093513, 0.0268531, -0.5182128, -0.1569117, -0.3492104, -0.2155812, -0.0392161, -0.3552701, 0.9319751, 0.1550327, -0.2962227, 0.9153284, 0.1470551, 0.3057349, 0.0617516, 0.5934935, -0.1118675, 0.6566533, -1.1410300, 0.7201911, 0.4613485, 0.4751079, -0.1956451, 0.0268744, -0.7405378, -0.0367488, 0.0942594, -0.5576359, 0.1901441, -0.4608253, 0.0174982, 0.2970467, 0.4230424, -0.1541975, 0.3828651, -0.1474487, -0.2809403, -0.0612479, 0.1860174, -0.0150790, -0.3129278, 0.3389230, 0.0144943, 0.2549954, -0.2039080, -0.1000422, -0.1439215, 0.1764802, -0.1757951, 0.3836617, -0.1316161, 0.0943417, 0.0123362, -0.2093474, 0.1063277, 0.2612825, -0.2790885, -0.0199187, 0.2368225, 0.5640559, 0.0145945, 0.1605127, 0.0941011, -0.3979308, 0.5283298, -0.3879965, 0.5292951, -0.0214575, 0.3128785, 0.1392319, -0.0357783, -0.2425334, 0.2766051, -0.1091767, 0.2472210, 0.0051574, -0.2188763, 0.1701127, 0.0324907, -0.0849292, 0.1132794, 0.1102249, 0.0826966, 0.2312904, -0.6955340, -0.3363871, 0.0778619, -0.3215622, -0.0235915, -0.0101222, -0.0934486, 0.1692782, -0.0411102, 0.3454086, -0.2621068, -0.0112024, 0.1919640, -0.2004677, 0.2520424, -0.0198838, 0.2598956, -0.0351781, 0.0903174, -0.0463146, -0.0826847, 0.0505923, -0.0602382, -0.0148637, -0.1614131, -0.1214503, 0.1613176, -0.1173977, 0.0996005, -0.1109961, 0.0922735, -0.0570295, 0.1194476, 0.0700215, -0.0606208, 0.1270892, 0.0712810, -0.2112693, -0.1931393, -0.1757720, 0.0951650, -0.3090639, -0.2786337, -0.3837940, 0.5698888, 0.1031060, 0.1700851, 0.2232326, -0.0569422, 0.1148025, 0.2036915, 0.1051866, 0.6243874, 0.6185188, -0.0999638, 0.0934461, -0.3487709, 0.5158741, 0.3924620, -0.1105984, 0.5366999, -0.8700936, -1.0222739, -0.1621883, 0.1432491, -0.0641485, -0.0606568, -0.5425753, -0.7233461, -0.6093550, 0.2249727, -0.3911178, 0.6345437, -0.0414115, -0.4625136, 0.2917506, 0.2109767, 0.6412838, 0.3307448, -0.7391987, -0.2074914, 0.3656350, 0.0904820, -0.2948250, -0.1879081, -0.2023569, 0.3664604, 0.2634708, -0.3531348, -0.6327595, 0.4403827, 0.2161390, 0.2077463, -0.3671280, 0.5333624, -0.1885908, 0.0684396, -0.0730250, 0.1153513, 0.3906485, -0.1929944, 0.1588487, -0.1496136, 0.0034723, 0.1058906, -0.0769879, 0.1061978, -0.0501986, -0.0496141, 0.0670391, -0.0377283, 0.0874361, 0.0650878, -0.0328690, -0.0664496, -0.6001179, 0.2986904, -0.2680899, 0.2626746, -0.1132197, 0.1229487, -0.0200973, 0.0313885, 0.6359798, -0.0922342, 0.1241876, -0.0888708, 0.0009254, 0.1357672, 0.0730435, -0.3235241, -0.1779824, -0.2997456, 0.2863789, -0.3487119, 0.1518145, -0.5578908, -0.2576382, -0.1182949, -0.4619887, 0.1600166, -0.1453047, 0.0680032, -0.2419467, 0.0001772, -0.1798283, -0.2129710, 0.9689248, -0.3833165, 0.0573459, -0.4214272, -0.0438824, 0.2583529, 0.5035083, -0.6810591, 0.1002424, 0.0361443, -0.3277960, -0.4970536, 0.2453915, -0.3229753, -0.1435948, -0.1736451, -0.7230493, -0.4691697, 0.2895883, -0.1456939, -0.1936812, -0.0954810, -0.1039214, -0.0166284, -0.2001947, -0.3186257, -0.1116009, 0.5662885, -0.2154126, 0.1714665, -0.4241474, 0.2451154, 0.7165399, -0.5589361, -0.1572717, 0.8552569, -0.2934281, -0.2107465, -0.2415778, 0.1111134, 0.0122270, 0.5683089, 0.0708979, 0.7062442, 0.4443365, -0.0249985, 0.7697493, 0.5327728, 0.3675256, -0.2004680, -0.7901496, -0.4140352, 0.1204381, 0.8648459, 0.0406502, -0.4981539, 0.3936258, -0.7343534, -0.4858570, -0.9478843, -0.4858515, 0.1701055, -0.4284994, -0.4669115, -0.6881824, -0.1255757, -0.4638782, 0.6190758, 0.4772179, 0.0836801, -0.0954533, 0.2589582, 0.6404762, -0.1024121, 0.2725188, -0.0567672, 0.1182317, 0.4446644, -0.3406713, 0.0559832, -0.4679064, -0.0479808, 0.0521079, -0.0870822, 0.0273100, -0.1110922, 0.0004704, -0.3015858, 0.0804126, -0.2845514, 0.0019115, -0.2926601, 0.0155520, -0.1964415, -0.0635939, -0.3067963, 0.3861094, -0.4183314, 0.1341913, -0.4031016, -0.2167615, 0.0837665, 0.3254414, -0.2497691, -0.1365314, 0.1347667, -0.1443004, 0.1201873, -0.0760484, -0.0813915, 0.2773890, -0.0225355, 0.2626073, -0.1171071, 0.2435607, -0.0927482, 0.1001449, -0.1702099, -0.3196163, 0.5877363, -0.0942353, 0.1096913, -0.0580086, 0.1658542, -0.1532428, 0.3838004, -0.1364820, 0.2379148, 0.0226721, -0.9486727, 0.0471871, -0.2889495, -0.0846890, 0.1797559, -0.2022166, -0.5794636, -0.1370407, -0.4106274, 0.2817611, -0.4343900, -0.6964189, 0.0995082, -0.0442456, -0.0315741, -0.0110493, 0.2104852, -0.3327667, -0.0426008, -0.1433302, 0.0624215, 0.0513770, 0.0806343, -0.2067210, -0.1681530, -0.9116020, 0.3187425, 0.0514263, 0.1780499, -0.0933935, -0.1283691, 0.2050675, 0.3542832, 0.5504524, 0.1712812, 0.3673798, -0.2179071, 0.0309926, 0.0976007, -0.1699147, 0.6405470, 0.6613162, 0.3034024, 0.7905031, -0.6097742, 0.4246245, 0.7306086, 0.5807211, -0.0064128, 0.3656771, -0.4617210, 0.2678794, -0.2004292, -0.3578294, -0.1726816, 0.0554052, 0.0350157, 0.5915754, -0.0203585, 0.0759241, -0.1140942, 0.1183311, -0.1510688, -0.0134705, -0.0978259, 0.2710258, -0.1428374, 0.2259427, -0.2612808, -0.1995445, 0.0324642, -0.2433879, -0.1193245, 0.0245992, -0.1118952, -0.1185197, -0.0737830, -0.1092648, -0.0074915, -0.0763595, 0.1441956, -0.1513248, 0.0880497, -0.1092044, 0.1980333, 0.2730721, -0.2452524, 0.3368914, 0.0111948, 0.0548908, 0.0059155, 0.0511906, -0.0759984, -0.0003335, 0.1121711, -0.4613860, 0.5134895, -0.3359855, 0.4769215, 0.0134460, 0.2828873, 0.2104597, -0.1148628, -0.2343793, 0.2841360, -0.3608468, 0.2986756, -0.0285679, -0.0970565, 0.0940803, 0.0710430, 0.3466266, -0.0252922, 0.0143420, 0.0146390, 0.0356186, 0.0324571, -0.1025298, 0.0905135, -0.1609289, 0.2104131, 0.4094425, 0.5369654, 0.3708864, -0.4917074, 0.2981741, 0.6447821, 0.0181734, -0.0014421, -0.2475360, 0.1685373, -0.0719392, -0.1766563, -0.0786047, 0.3887443, 0.5683194, 0.1227366, -0.3889403, 0.2959919, -0.0507108, 0.3383470, -0.0792183, -0.2859499, -0.1036216, 0.1382751, -0.1976376, 0.2762082, 0.3267169, -0.4594756, 0.0598829, 0.2913103, -0.3694354, 0.3737148, -0.5183859, -0.7008716, -0.0395234, 0.4852855, -0.2171500, -0.7053072, -0.4991091, 0.2024119, 0.1592132, -0.4152100, -0.4256546, 0.3087745, -0.0806829, -0.0935526, 0.0294502, -0.1627689, -0.4609432, 0.5260309, -0.6456886, 0.2007331, 0.2933916, -0.1396879, -0.0415259, -0.0670704, 0.0115323, 0.1782231, -0.0767321, -0.1105404, -0.2647381, -0.0274911, -0.0698640, -0.0038630, -0.0970322, -0.0603650, -0.0857060, 0.2031547, 0.1831581, -0.4518503, 0.1654684, -0.1426963, -0.1503477, -0.1393246, -0.2026948, 0.0429422, 0.2370295, -0.1831069, 0.3895363, -0.0695245, 0.0433065, -0.0564393, -0.1417701, 0.2155443, -0.2995106, 0.3182778, 0.5450820, -0.2252713, 0.0363688, -0.1718405, -0.2620012, 0.4342357, 0.2754403, -0.0221255, 0.0961393, 0.1443960, 0.1017210, 0.1325648, 0.1425537, -0.1785801, -0.3250968, 0.0754474, -0.3749839, -0.0162527, -0.0094159, -0.0786251, 0.0378912, 0.0945599, 0.1461810, -0.3464882, 0.3093731, -0.3585347, 0.2326562, -0.3336061, -0.1460380, -0.2435825, -0.0401674, 0.1928711, 0.0807574, -0.2018481, -0.4270057, -0.2849124, -0.3355846, 0.3214097, -0.3571669, -0.2502927, -0.0129574, 0.0189537, -0.4385507, 0.2072561, 0.0513688, 0.2988456, -0.0501798, -0.0621900, 0.1694670, -0.2064065, 0.0896689, 0.7913690, 0.3585398, -0.0817096, 0.0598960, 0.1050461, 0.2475841, -0.6490353, 0.2523943, -0.2230127, -0.0029347, 0.0097553, -0.4517760, -0.3046168, 0.1346664, -0.1032889, 0.7216869, 0.6028582, 0.4497806, -0.2393321, 0.1755342, 0.1653123, -0.0413742, 0.2623122, -0.1941507, -0.6127819, -0.1472983, 0.1895358, 0.0048919, 0.0290338, 0.0057643, 0.0785637, -0.0767744, 0.4029086, 0.5604403, -0.1909199, 0.3173218, 0.3480718, 0.4091369, -0.3377852, 0.3681487, 0.0921967, 0.2124878,
		})
		actual := c.Apply(in)
		if expected.Height != actual.Height || expected.Width != actual.Width ||
			expected.Depth != actual.Depth {
			t.Fatal("incorrect output shape")
		}
		for i, x := range expected.Data {
			a := actual.Data[i]
			if math.Abs(float64(x-a)) > 1e-4 {
				t.Errorf("bad value at %d: expected %f but got %f", i, x, a)
			}
		}
	}

	p := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(p)

	runtime.GOMAXPROCS(1)
	t.Run("Proc1", runTest)

	runtime.GOMAXPROCS(2)
	t.Run("Proc2", runTest)
}
