--creacion de tabla de cantidad de seguidores por usuario

select id_usuario,count(id_usuario_seguido) as cant_seguidores
into usuarios_seguidores
from siguiendo
group by id_usuario;

--tabla de datos de restaurantes

select id_restaurante,localidad,cocina,precio,
	   case when telefono is not null then 1 else 0 end as telefono,
       precio, 
       case when char_length(rating_comida)>4 then 
       cast(substr(rating_comida,1,2) as numeric)/30 
       else cast(substr(rating_comida,1,1) as numeric)/30 end comida_oleo,
       case when char_length(rating_servicio)>4 then 
       cast(substr(rating_servicio,1,2) as numeric)/30 
       else cast(substr(rating_servicio,1,1) as numeric)/30 end servicio_oleo,
       case when char_length(rating_ambiente)>4 then 
       cast(substr(rating_ambiente,1,2) as numeric)/30 
       else cast(substr(rating_ambiente,1,1) as numeric)/30 end ambiente_oleo,
       fotos
into rest_campos
from restaurantes