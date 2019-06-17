import AZ 
import Utilities.GameRules1 as gm
import torch

def create_agent(PATH):
	agente1 = AZ.AgentZero('cuda')
	agente1.net.load_state_dict(torch.load(PATH))
	agente1.net.eval()

	return agente1






if __name__ =="__main__":
	PATH = "/home/dell/Descargas/AlphaZero/Models/Model_4_3.pth"
	PATH2  = "/home/dell/Descargas/AlphaZero/Models/Model_6_5.pth"
	
	agente1 = create_agent(PATH)
	agente2 = create_agent(PATH2)
	#Ver quien gana en porcentaje.
	print(AZ.evaluate(agente2,agente1))
	
	
	"""
	tablero = gm.BoardGame1(6,7)
	reglas = gm.conecta4()
	accion = [-1,-1,-1]
	jugador_turno = 1
 
	while not reglas.is_complete(tablero,accion):
		if jugador_turno == 1:
			accion,reward = agente1.take_action(tablero)
			accion[2] = jugador_turno
			print(accion)
			print(reward)
			reglas.execute_action(tablero,accion)
			jugador_turno = 2
			continue
		
	
		nodo = int(input())
		accion = [tablero.stones_stack[nodo] + 1 , nodo, 2]
		reglas.execute_action(tablero,accion)
		jugador_turno = 1
	"""
	
