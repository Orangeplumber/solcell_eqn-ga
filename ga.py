import numpy as np
from math import *

def FeatureCalc(before_dec_binary,after_dec_binary):
	# will return a decimel value of a feature
	before_dec_integer=0
	j=len(before_dec_binary)
	for i in range(len(before_dec_binary)):
		j-=1
		before_dec_integer+=before_dec_binary[i]*pow(2,j)

	after_dec_integer=0.00000000
	j=len(after_dec_binary)
	for i in range(len(after_dec_binary)):
		j-=1
		after_dec_integer+=after_dec_binary[i]*pow(2,j)

	while after_dec_integer>=1:
		after_dec_integer=after_dec_integer/10

	feature=before_dec_integer+after_dec_integer
	return feature


class GenerationSimulator:
	population=0
	mutation=0
	people=[]

	def __init__(self,population,data):
		self.fitness=[]
		self.data=data
		self.population=population
		for i in range(population):
			features={'gen':0,'sat_current':np.random.choice([0, 1],size=(24,),p=[1./3, 2./3]),'Io':np.random.choice([0, 1],size=(24,),p=[1./3, 2./3]),'Rse':np.random.choice([0, 1],size=(24,),p=[1./3, 2./3]),'Rsh':np.random.choice([0, 1],size=(24,),p=[1./3, 2./3]),'N':np.random.choice([0, 1],size=(24,),p=[1./3, 2./3]),'N_int':np.random.choice([0, 1],size=(3,),p=[1./3, 2./3]),'sat_current_int':np.random.choice([0, 1],size=(2,),p=[1./3, 2./3]),'Io_int':np.random.choice([0, 1],size=(2,),p=[1./3, 2./3]),'Rse_int':np.random.choice([0, 1],size=(6,),p=[1./3, 2./3]),'Rsh_int':np.random.choice([0, 1],size=(1,),p=[1./3, 2./3])}
			error=self.ErrorCalc(features)
			features['error']=error
			self.people.append(features)
		self.gen=0


	def GenerationEpoch(self,crossover,mutation):
		# calculation of fitness of each person
		# will return the ten person with lowest errors
		# sorting
		self.gen+=1
		for i in range(self.population):
			for j in range(i+1,self.population):
				if self.people[i]['error']>self.people[j]['error']:
					temp=self.people[j]
					self.people[j]=self.people[i]
					self.people[i]=temp
		# crossover
		r=crossover/2
		survival=int((1-crossover)*self.population)
		self.new_people=self.people[:survival]
		crossover_population=self.population-survival
		for (p1,p2) in zip(range(crossover_population),reversed(range(crossover_population))):
			self.new_people.append(self.CrossOverFunc(p1,p2))

		# mutation
		mutation_population=int(mutation*self.population)
		random_mutation=np.random.random_integers(0,self.population-1,size=(mutation_population,))
		for i in random_mutation:
			self.MutationFunc(i)

		performers=self.people[:10]
		self.people=self.new_people
		return performers

	def Flipping(self,person,feature,flipping_bit):
		if self.new_people[person][feature][flipping_bit]==1:
			self.new_people[person][feature][flipping_bit]=0
		else:
			self.new_people[person][feature][flipping_bit]=1

	def MutationFunc(self,person):
		flipping_bits=np.random.random_integers(0,23,size=5)
		for i in flipping_bits:
			self.Flipping(person,'sat_current',i)
			self.Flipping(person,'Io',i)
			self.Flipping(person,'Rse',i)
			self.Flipping(person,'Rsh',i)
			self.Flipping(person,'N',i)
		changing_feature=np.random.random_integers(0,4,size=1)
		if changing_feature[0]==0:
			flipping_bits=np.random.random_integers(0,1,size=1)
			self.Flipping(person,'sat_current_int',flipping_bits[0])
		if changing_feature[0]==1:
			flipping_bits=np.random.random_integers(0,1,size=1)
			self.Flipping(person,'Io_int',flipping_bits[0])
		if changing_feature[0]==2:
			flipping_bits=np.random.random_integers(0,5,size=1)
			self.Flipping(person,'Rse_int',flipping_bits[0])
		if changing_feature[0]==3:
			self.Flipping(person,'Rsh_int',0)
		if changing_feature[0]==4:
			flipping_bits=np.random.random_integers(0,2,size=1)
			self.Flipping(person,'N',flipping_bits[0])
		

	def CrossOverFunc(self,p1,p2):
		gen=self.gen
		sat_current=np.column_stack((self.people[p1]['sat_current'][:12],self.people[p2]['sat_current'][12:])).T.flatten()
		Io=np.column_stack((self.people[p1]['Io'][:12],self.people[p2]['Io'][12:])).T.flatten()
		N=np.column_stack((self.people[p1]['N'][:12],self.people[p2]['N'][12:])).T.flatten()
		Rse=np.column_stack((self.people[p1]['Rse'][:12],self.people[p2]['Rse'][12:])).T.flatten()
		Rsh=np.column_stack((self.people[p1]['Rsh'][:12],self.people[p2]['Rsh'][12:])).T.flatten()
		sat_current_int=np.column_stack((self.people[p1]['sat_current_int'][:1],self.people[p2]['sat_current_int'][1:])).T.flatten()
		Io_int=np.column_stack((self.people[p1]['Io_int'][:1],self.people[p2]['Io_int'][1:])).T.flatten()
		Rse_int=np.column_stack((self.people[p1]['Rse_int'][:3],self.people[p2]['Rse_int'][3:])).T.flatten()
		Rsh_int=self.people[p1]['Rsh_int']
		N_int=self.people[p2]['N_int']
		person={'gen':gen,'sat_current':sat_current,'Io':Io,'Rse':Rse,'Rsh':Rsh,'Rsh_int':Rsh_int,'Io_int':Io_int,'Rse_int':Rse_int,'sat_current_int':sat_current_int,'N':N,'N_int':N_int}
		error=self.ErrorCalc(person)
		person['error']=error
		return person

	def ErrorCalc(self,person):
		# will return total error in the performance of a person
		sat_current=FeatureCalc(person['sat_current_int'],person['sat_current'])
		Io=FeatureCalc(person['Io_int'],person['Io'])
		Rse=FeatureCalc(person['Rse_int'],person['Rse'])
		Rsh=FeatureCalc(person['Rsh_int'],person['Rsh'])
		N=FeatureCalc(person['N_int'],person['N'])
		error=0
		for i in range(self.data.shape[0]):
			error+=self.ErrorFunc(sat_current,Io,Rse,Rsh,N,i)
		return error

	def ErrorFunc(self,sat_current,Io,Rse,Rsh,N,i):
		# will return the error in one data
		# sat_current is in ampere
		# Io is in micro-ampere
		# N is 1-5
		# Rse is 0-500
		# Rsh is 0-1000
		# precision is upto 7 didit upto
		if N==0:
			N=np.random.random_integers(1,7,size=1)[0]
		if Rsh==0:
			Rsh=0.000001
		I=self.data[i][1]
		V=self.data[i][0]
		target=I
		K=1.38064852*pow(10,23)
		T=298
		exponent=9*(V+I*Rse)/(N*K*T)
		output=sat_current-Io*(exp(exponent)-1)*pow(10,-6)-((V+I*Rse)/(Rsh*1000))
		return abs(target-output)

if __name__=="__main__":
	print "Calculation of constants of solar cell equation!"
	data=np.genfromtxt('data.csv',delimiter=",")
	ga=GenerationSimulator(1000,data)
	for i in range(100):
		print "Generation:", i
		performers=ga.GenerationEpoch(0.95,0.5)
		print performers[0]['error']
	print performers[0]