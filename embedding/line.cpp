#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000; // 应该大于最大节点数
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;

struct ClassVertex{
    double degree;
    char *name;
};

char network_file[MAX_STRING],embedding_file[MAX_STRING];
struct ClassVertex *vertex;
int is_binary=0,num_threads=1,order=2,dim=100,num_negative=5;
int *vertex_hash_table,*neg_table;
int max_num_vertices=1000,num_vertices=0;
long long total_samples=1,current_sample_count=0,num_edges=0;
real init_rho=0.025,rho;
real *emb_vertex,*emb_context,*sigmoid_table;

int *edge_source_id,*edge_target_id;
double *edge_weight;

long long *alias;
double *prob;

const gsl_rng_type*gsl_T;
gsl_rng *gsl_r;


// hash
unsigned int Hash(char *key){
    unsigned int seed = 131;
    unsigned int hash = 0;
    while (*key){
        hash =hash*seed + (*key++);
    }
    return hash % hash_table_size;
}

void InitHashTable(){
    vertex_hash_table = (int *)malloc(hash_table_size*sizeof(int));
    for (int k=0;k != hash_table_size; k++){
        vertex_hash_table[k]=-1;
    }
}

// 插入
int InsertHashTable(char *key, int value){
    int addr = Hash(key);
    while (vertex_hash_table[addr] != -1) addr=(addr+1)%hash_table_size;
    vertex_hash_table[addr] = value;
}

// hash冲突解决方法，该函数判断key是否已经被指定，-1表示没有被指定
int SearchHashTable(char *key){
    int addr = Hash(key);
    while(1){
        if (vertex_hash_table[addr] == -1)return -1; // 等于-1表示该位置没有被占用，可以被使用
        if (! strcmp(key,vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr]; // 等于0表示已被占用且为该key，返回被占用的值
        addr = (addr+1)%hash_table_size;  // 不等于0表示被占用且不是该key，从该位置找下一个值
    }
    return -1;
}

// index
int AddVertex(char *name){
    int length=strlen(name)+1;
    if (length > MAX_STRING) length=MAX_STRING;
    vertex[num_vertices].name=(char *)calloc(length,sizeof(char));
    strncpy(vertex[num_vertices].name,name,length-1);
    vertex[num_vertices].degree=0;
    num_vertices++;
    // 动态增加地址
    if (num_vertices+2 >= max_num_vertices){
        max_num_vertices += 1000;
        vertex = (struct ClassVertex *)realloc(vertex,max_num_vertices*sizeof(struct ClassVertex));
    }
    // 插入
    InsertHashTable(name,num_vertices-1);
    return num_vertices-1;
}

void ReadData(){
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2*MAX_STRING+10000];
    int vid;
    double weight;

    // 打开二进制文件，可读
    fin=fopen(network_file,"rb");
    if(fin == NULL){
        printf("ERROR: network file not found!\n");
        exit(1);
    }

    num_edges =0;
    // char *fgets(char *s, int size, FILE *stream)
    while (fgets(str,sizeof(str),fin)) num_edges++;
    fclose(fin);
    // l 表示长整形或双精度浮点数
    // d 表示有符号10进制整数
    printf("Numberr of edges: %lld     \n", num_edges);

    edge_source_id=(int *)malloc(num_edges*sizeof(int));
    edge_target_id=(int *)malloc(num_edges*sizeof(int));
    edge_weight=(double *)malloc(num_edges*sizeof(double));

    if(edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL){
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    fin=fopen(network_file,"rb");
    num_vertices=0;
    for(int k=0; k!=num_edges;k++){
        fscanf(fin,"%s %s $lf",name_v1,name_v2,&weight);
        if (k % 10000 ==0){
            printf("Reading edges: %.3lf%%c",k/(double)(num_edges+1)*100,13);
            fflush(stdout);
        }
        vid =SearchHashTable(name_v1);
        if (vid == -1) vid = AddVertex(name_v1);
        vertex[vid].degree+=weight; //degree表示的是节点的出边权重，那么line是不区分有向和无向的，默认一条边两边都有箭头
        edge_source_id[k]=vid;

        vid=SearchHashTable(name_v2);
        if (vid == -1) vid=AddVertex(name_v2);
        vertex[vid].degree += weight;
        edge_target_id[k]=vid;

        edge_weight[k]=weight;
    }
    fclose(fin);
    printf("Number of vertices: %d \n",num_vertices);
}


// 判断参数中是否有给定的字符串，有则返回下标
int ArgPos(char *str, int argc, char const *argv[]){
    int a;
    for(a=1;a<argc;a++){
        if (!strcmp(str,argv[a])){
            if (a == argc -1){
                printf("Argument missing for %s\n",str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

// https://blog.csdn.net/haolexiao/article/details/65157026
// https://shomy.top/2017/05/09/alias-method-sampling/
void InitAliasTable(){
    //构建好下面两个数组后，采样的过程分两步，第一步是均匀采样在0-N中(N在本代码中是num_edges)，结果为k，第二步均匀采样[0-1]中，小于prob[k]则结果为k，反之为alias[k]
    alias = (long long *)malloc(num_edges*sizeof(long long)); // 每一列第二层的类型
    prob = (double *)malloc(num_edges*sizeof(double)); // 落在原类型的概率，即每一列第一层的概率
    if (alias == NULL || prob == NULL){
        printf("Error: memory allocation failed! \n");
        exit(1);
    }
    double sum=0;
    long long cur_small_block,cur_large_block;
    long long num_small_block=0,num_large_block =0;

    for (long long k=0;k!=num_edges;k++)sum+=edge_weight[k]; //计算全部权重的和
    for (long long k=0;k!=num_edges;k++)norm_prob[k]=edge_weight[k]*num_edges/sum;  //alias中的N=num_edges，【全局归一化应该是为了保证和为1，便于alias采样】

    // 小于1的放到small_block中，大于1的放到large_block中
    for (long long k =num_edges -1;k>=0;k--){
        if(norm_prob[k]<1){
            samll_block[num_small_block++] = k;
        }else{
            large_block[num_large_block++] = k;
        }
    }

    // 只要num_small_block或num_large_block中有一项为0，则其它的就全为1了
	while (num_small_block && num_large_block){
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block]; // 第1层的概率
		alias[cur_small_block] = cur_large_block; // 第2层的颜色
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1; //large剩下的部分
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

//初始化目标矩阵和上下文矩阵
void InitVector(){
	long long a, b;
    // 用法类似malloc，第三个参数是开辟的空间，第一个参数是将空间地址返回给emb_vertex
	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
}

// 负采样，对table填充
void InitNegTable(){
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

// 初始化sigmoid
void InitSigmoidTable(){
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x){
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

int Rand(unsigned long long &seed){
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label){
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

long long SampleAnEdge(double rand_value1, double rand_value2){
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

void *TrainLINEThread(void *id){
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r)); // 返回的是边的index
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim;
			if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
			if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);
		}
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void TrainLINE(){
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (order != 1 && order !=2){
        printf("Error:order should be either 1 or 2! \n");
        exit(1);
    }
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

    InitHashTable(); //初始化hashtable
    ReadData(); //读取边数据
    InitAliasTable(); // 初始化alias数组
    InitVector(); //初始化embedding
	InitNegTable(); //初始化neg table
	InitSigmoidTable(); //初始忽视sigmoid

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();   

}



// main 函数中传入的两个参数，一个是argument count，一个是argument vector 分别表示参数的个数和参数数组,argv[0]表示程序的名称
int main(int argc, char const *argv[]){
    int i;
    if (argc == 1){
		printf("LINE: Large Information Network Embedding\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse network data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the learnt embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
    }

    if ((i=ArgPos((char *)"-train",argc,argv))>0) strcpy(network_file,argv[i+1]);
    if ((i=ArgPos((char *)"-output",argc,argv))>0) strcpy(embedding_file,argv[i+1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    total_samples *= 1000000;
    rho = init_rho;
    vertex = (struct ClassVertex *)calloc(max_num_vertices,sizeof(struct ClassVertex));
    TrainLINE();
    return 0;
}
