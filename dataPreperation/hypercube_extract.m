hyper_dir = "Catalogs/organic_non_organic_apples/nonorganic/";

%k = [094 095 096 097 098 099];
%k = [100	101	102	103	104	105	106	107	108	109	110	111	112	113	114	115	116	117	118	119	120	121	122	123];
%k = [13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38	39];
%k = [41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56];
%k = [125	126	127	128	129	130	131	132	133	134	135	136	137	138];
%k = [58	59	60	61	62	63	64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84	85	86	87	88	89	90	91	92];
k = 235;
for i = 1 : size(k, 2)
    h = [hyper_dir num2str(k(i)) "/results/" "REFLECTANCE_" num2str(k(i)) ".hdr"];
    hcube = hypercube(h);
    rad = hcube.DataCube;
    matfile = ["working_organic/working_nonorganic_204ch/" num2str(k(i))];
    %save(matfile, "rad");
end